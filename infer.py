import os
import sys
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import paddle
import paddle.nn as nn
import paddle.nn.functional as PF
from utils.utils_gcdrnet import *
from models.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L
from models.sa_aidr import STRAIDR
import gc


class CombinedProcessor:
    def __init__(self):
        # 设备设置
        self.torch_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        if paddle.is_compiled_with_cuda():
            paddle.set_device('gpu:1')
        else:
            paddle.set_device('cpu')

        print(f"外观增强推理设备: {self.torch_device}")
        print(f"手写擦除推理设备：{paddle.device.get_device()}")
        self.load_models()

    def load_models(self):
        print("加载外观增强模型...")

        # 外观增强模型
        self.gcnet = UNext_full_resolution_padding(
            num_classes=3, input_channels=3, img_size=512
        ).to(self.torch_device)

        self.drnet = UNext_full_resolution_padding_L_py_L(
            num_classes=3, input_channels=6, img_size=512
        ).to(self.torch_device)

        # 加载权重
        gcnet_state = convert_state_dict(
            torch.load('./ckpts/gcdr_net/gcnet/checkpoint.pkl',
                       map_location=self.torch_device)['model_state']
        )
        self.gcnet.load_state_dict(gcnet_state)
        self.gcnet.eval()

        drnet_state = convert_state_dict(
            torch.load('./ckpts/gcdr_net/drnet/checkpoint.pkl',
                       map_location=self.torch_device)['model_state']
        )
        self.drnet.load_state_dict(drnet_state)
        self.drnet.eval()

        print("加载手写擦除模型...")
        self.netG = STRAIDR(num_c=96)
        weights = paddle.load("./ckpts/STE_idr_best.pdparams")
        self.netG.load_dict(weights)
        self.netG.eval()
        print("模型加载完成")

    def clear_gpu_memory(self):
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Paddle的内存清理
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
        gc.collect()

    def appearance_enhancement(self, image):
        im_org, padding_h, padding_w = stride_integral(image)
        h, w = im_org.shape[:2]

        with torch.no_grad():
            im = torch.from_numpy(im_org.transpose(2, 0, 1) / 255).unsqueeze(0)
            im = im.float().to(self.torch_device)
            im_org_tensor = torch.from_numpy(im_org.transpose(2, 0, 1) / 255).unsqueeze(0)
            im_org_tensor = im_org_tensor.float().to(self.torch_device)

            shadow = self.gcnet(im)
            shadow = F.interpolate(shadow, (h, w))

            model1_im = torch.clamp(im_org_tensor / shadow, 0, 1)
            pred, _, _, _ = self.drnet(torch.cat((im_org_tensor, model1_im), 1))

            pred = pred[0].permute(1, 2, 0).data.cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            pred = pred[padding_h:, padding_w:]

            # 清理中间变量
            del im, im_org_tensor, shadow, model1_im

        return pred

    def handwriting_erasing(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = paddle.to_tensor(
            image_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        ).unsqueeze(0)

        _, _, h, w = image_tensor.shape

        if h < 1600 and w < 1600:
            pad_size = 128
            h_padded = w_padded = False

            if h % pad_size != 0:
                pad_h = pad_size - (h % pad_size)
                image_tensor = PF.pad(image_tensor, (0, 0, 0, pad_h), mode='reflect')
                h_padded = True

            if w % pad_size != 0:
                pad_w = pad_size - (w % pad_size)
                image_tensor = PF.pad(image_tensor, (0, pad_w, 0, 0), mode='reflect')
                w_padded = True

            with paddle.no_grad():
                res = self.netG(image_tensor)[3]
                res += paddle.flip(self.netG(paddle.flip(image_tensor, axis=[2]))[3], axis=[2])
                res += paddle.flip(self.netG(paddle.flip(image_tensor, axis=[3]))[3], axis=[3])
                res += paddle.flip(self.netG(paddle.flip(image_tensor, axis=[2, 3]))[3], axis=[2, 3])
                res = res / 4

            if h_padded:
                res = res[:, :, 0:h, :]
            if w_padded:
                res = res[:, :, :, 0:w]
        else:
            pad = 112
            m = nn.Pad2D(pad, mode='reflect')
            image_tensor = m(image_tensor)

            _, _, h_new, w_new = image_tensor.shape
            step = 800
            res = paddle.zeros_like(image_tensor)

            for i in range(0, h_new, step):
                for j in range(0, w_new, step):
                    if h_new - i < step + 2 * pad:
                        i = h_new - (step + 2 * pad)
                    if w_new - j < step + 2 * pad:
                        j = w_new - (step + 2 * pad)

                    clip = image_tensor[:, :, i:i + step + 2 * pad, j:j + step + 2 * pad]

                    with paddle.no_grad():
                        g_images_clip = self.netG(clip)[3]
                        g_images_clip += paddle.flip(self.netG(paddle.flip(clip, axis=[2]))[3], axis=[2])
                        g_images_clip += paddle.flip(self.netG(paddle.flip(clip, axis=[3]))[3], axis=[3])
                        g_images_clip += paddle.flip(self.netG(paddle.flip(clip, axis=[2, 3]))[3], axis=[2, 3])
                        g_images_clip = g_images_clip / 4

                    res[:, :, i + pad:i + step + pad, j + pad:j + step + pad] = \
                        g_images_clip[:, :, pad:-pad, pad:-pad]
                    
                    # 大图切块处理时清理内存
                    del clip, g_images_clip
                    self.clear_gpu_memory()

            res = res[:, :, pad:-pad, pad:-pad]

        # 转换回图像
        img = res.squeeze().cpu().numpy()
        img = img.clip(0, 1)
        img = (img * 255.0)
        img = np.transpose(img, (1, 2, 0))
        img = img.round()
        img = img[:, :, ::-1]
        
        # 清理paddle变量
        del image_tensor, res
        
        return img.astype(np.uint8)

    def process_single(self, input_path, output_path):
        print(f"处理图片: {input_path}")

        image = cv2.imread(input_path)
        if image is None:
            print(f"无法读取图片: {input_path}")
            return False

        start_time = time.time()

        # 外观增强
        enhanced_image = self.appearance_enhancement(image)
        
        # 清理内存
        self.clear_gpu_memory()

        # 手写擦除
        final_result = self.handwriting_erasing(enhanced_image)
        
        # 清理内存
        self.clear_gpu_memory()

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, final_result)

        if success:
            total_time = time.time() - start_time
            print(f"处理完成，耗时: {total_time:.2f}秒，保存至: {output_path}")
            return True
        else:
            print(f"保存失败: {output_path}")
            return False

    def process_batch(self, input_dir, output_dir):
        print(f"批量处理目录: {input_dir}")

        # 获取图片文件
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp')
        image_files = []

        for file in os.listdir(input_dir):
            if file.lower().endswith(supported_formats):
                image_files.append(file)

        if not image_files:
            print("目录中没有找到支持的图片文件")
            return

        print(f"找到 {len(image_files)} 张图片")
        os.makedirs(output_dir, exist_ok=True)

        success_count = 0
        start_time = time.time()

        for i, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"[{i + 1}/{len(image_files)}] {filename}")

            if self.process_single(input_path, output_path):
                success_count += 1
            
            # 每处理一张图片后强制清理内存
            self.clear_gpu_memory()

        total_time = time.time() - start_time
        print(f"批量处理完成: {success_count}/{len(image_files)} 成功")
        print(f"总耗时: {total_time:.2f}秒，平均: {total_time / len(image_files):.2f}秒/张")


def main():
    parser = argparse.ArgumentParser(description='外观增强 + 手写擦除处理')
    parser.add_argument('--input', '-i', default="./images/", help='输入路径 (图片文件或文件夹)')
    parser.add_argument('--output', '-o', default="./results", help='输出路径')

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"输入路径不存在: {input_path}")
        sys.exit(1)

    processor = CombinedProcessor()

    if os.path.isfile(input_path):
        # 单张图片处理 - 修复路径问题
        input_filename = os.path.basename(input_path)
        input_name, input_ext = os.path.splitext(input_filename)
        
        # 如果输出路径是目录，则在目录下创建对应的文件名
        if os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
            # 输出路径是目录或没有扩展名，视为目录
            os.makedirs(output_path, exist_ok=True)
            output_file_path = os.path.join(output_path, f"processed_{input_filename}")
        else:
            # 输出路径是文件路径
            output_file_path = output_path
        
        print(f"输入: {input_path}")
        print(f"输出: {output_file_path}")
        processor.process_single(input_path, output_file_path)
        
    elif os.path.isdir(input_path):
        # 批量处理
        processor.process_batch(input_path, output_path)
    else:
        print(f"无效的输入路径: {input_path}")


if __name__ == "__main__":
    main()