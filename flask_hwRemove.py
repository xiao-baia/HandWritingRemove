import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import cv2
import time
import uuid
import base64
import tempfile
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import paddle
import paddle.nn as nn
import paddle.nn.functional as PF
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from utils.utils_gcdrnet import *
from models.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L
from models.sa_aidr import STRAIDR
import gc

app = Flask(__name__)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# 全局处理器实例
processor = None


class CombinedProcessor:
    def __init__(self, torch_device_id=0, paddle_device_id=0):
        # 设备设置 - 在创建任何模型之前设置默认CUDA设备
        self.torch_device_id = torch_device_id
        self.paddle_device_id = paddle_device_id

        # 设置PyTorch默认设备
        if torch.cuda.is_available() and torch_device_id >= 0:
            torch.cuda.set_device(torch_device_id)  # 重要：先设置默认设备
            self.torch_device = torch.device(f"cuda:{torch_device_id}")
            # 设置cuDNN确定性行为，避免使用其他GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            self.torch_device = torch.device("cpu")

        # 设置PaddlePaddle设备
        if paddle.is_compiled_with_cuda() and paddle_device_id >= 0:
            paddle.set_device(f'gpu:{paddle_device_id}')
        else:
            paddle.set_device('cpu')

        print(f"外观增强推理设备: {self.torch_device}")
        print(f"手写擦除推理设备：{paddle.device.get_device()}")

        self.load_models()

    def load_models(self):
        print("加载外观增强模型...")

        # 确保在正确的设备上创建模型
        with torch.cuda.device(self.torch_device_id if torch.cuda.is_available() else None):
            # 外观增强模型 - 直接在目标设备上创建
            self.gcnet = UNext_full_resolution_padding(
                num_classes=3, input_channels=3, img_size=512
            )
            # 立即移动到目标设备
            self.gcnet = self.gcnet.to(self.torch_device)

            self.drnet = UNext_full_resolution_padding_L_py_L(
                num_classes=3, input_channels=6, img_size=512
            )
            self.drnet = self.drnet.to(self.torch_device)

            # 加载权重时使用map_location确保权重直接加载到目标设备
            gcnet_checkpoint = torch.load(
                './ckpts/gcdr_net/gcnet/checkpoint.pkl',
                map_location=self.torch_device
            )
            gcnet_state = convert_state_dict(gcnet_checkpoint['model_state'])
            self.gcnet.load_state_dict(gcnet_state)
            self.gcnet.eval()

            drnet_checkpoint = torch.load(
                './ckpts/gcdr_net/drnet/checkpoint.pkl',
                map_location=self.torch_device
            )
            drnet_state = convert_state_dict(drnet_checkpoint['model_state'])
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
            with torch.cuda.device(self.torch_device_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
        gc.collect()

    def appearance_enhancement(self, image):
        # 确保在正确的设备上下文中执行
        with torch.cuda.device(self.torch_device_id if torch.cuda.is_available() else None):
            im_org, padding_h, padding_w = stride_integral(image)
            h, w = im_org.shape[:2]
            im = im_org

            with torch.no_grad():
                # 创建tensor时直接指定设备
                im = torch.from_numpy(im_org.transpose(2, 0, 1) / 255).unsqueeze(0)
                im = im.float().to(self.torch_device, non_blocking=True)

                im_org_tensor = torch.from_numpy(im_org.transpose(2, 0, 1) / 255).unsqueeze(0)
                im_org_tensor = im_org_tensor.float().to(self.torch_device, non_blocking=True)

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

    def process_image(self, image_data, use_enhancement=True, additional_erase_count=0):
        """处理图像并返回结果"""
        try:
            start_time = time.time()

            enhanced_image = None
            first_erase_result = None
            final_result = None

            if use_enhancement:
                # 第一步：外观增强
                enhanced_image = self.appearance_enhancement(image_data)
                self.clear_gpu_memory()
                print(f"外观增强完成")

                # 第二步：第一次擦除
                first_erase_result = self.handwriting_erasing(enhanced_image)
                self.clear_gpu_memory()
                print(f"第一次擦除完成")

                # 第三步：额外擦除次数
                current_image = first_erase_result.copy()
                for i in range(additional_erase_count):
                    current_image = self.handwriting_erasing(current_image)
                    self.clear_gpu_memory()
                    print(f"第{i+2}次擦除完成")

                final_result = current_image
            else:
                # 直接进行手写擦除，跳过外观增强
                first_erase_result = self.handwriting_erasing(image_data)
                self.clear_gpu_memory()
                print(f"第一次擦除完成（无外观增强）")

                # 额外擦除次数
                current_image = first_erase_result.copy()
                for i in range(additional_erase_count):
                    current_image = self.handwriting_erasing(current_image)
                    self.clear_gpu_memory()
                    print(f"第{i+2}次擦除完成")

                final_result = current_image

            total_time = time.time() - start_time
            print(f"图像处理完成，耗时: {total_time:.2f}秒")

            return True, final_result, enhanced_image, first_erase_result, total_time

        except Exception as e:
            print(f"处理图像时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, str(e), None, None, 0


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def init_processor(torch_device_id=0, paddle_device_id=0):
    """初始化处理器"""
    global processor
    if processor is None:
        try:
            processor = CombinedProcessor(torch_device_id, paddle_device_id)
            print("处理器初始化成功")
        except Exception as e:
            print(f"处理器初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        if processor is None:
            return jsonify({
                'status': 'error',
                'message': '模型未初始化'
            }), 500

        return jsonify({
            'status': 'ok',
            'message': '服务正常',
            'torch_device': str(processor.torch_device),
            'paddle_device': paddle.device.get_device()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/process', methods=['POST'])
def process_image():
    """图像处理接口 - 返回base64编码的处理结果"""
    try:
        # 检查是否有文件
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '没有上传文件'
            }), 400

        file = request.files['image']

        # 检查文件名
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': '没有选择文件'
            }), 400

        # 检查文件类型
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': f'不支持的文件类型，支持的格式: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400

        # 获取处理模式参数
        use_enhancement = request.form.get('use_enhancement', 'true').lower() == 'true'
        additional_erase_count = int(request.form.get('additional_erase_count', '0'))

        # 验证额外擦除次数
        if additional_erase_count < 0 or additional_erase_count > 10:
            return jsonify({
                'status': 'error',
                'message': '额外擦除次数必须在0-10之间'
            }), 400

        # 直接读取文件到内存
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                'status': 'error',
                'message': '无法解析图像文件'
            }), 400

        print(f"开始处理图像，尺寸: {image.shape}, 使用外观增强: {use_enhancement}, 额外擦除次数: {additional_erase_count}")

        # 处理图像
        success, final_result, enhanced_image, first_erase_result, processing_time = processor.process_image(
            image, use_enhancement, additional_erase_count)

        if success:
            # 将结果编码为base64
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            if file_extension == 'png':
                encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
            else:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

            response = {
                'status': 'success',
                'format': file_extension,
                # 'use_enhancement': use_enhancement,
                # 'additional_erase_count': additional_erase_count,
                'total_erase_count': 1 + additional_erase_count,
                #'processing_time': f"{processing_time:.2f}秒"
            }

            # 编码最终结果
            if file_extension == 'png':
                success_encode, buffer = cv2.imencode('.png', final_result, encode_param)
            else:
                success_encode, buffer = cv2.imencode('.jpg', final_result, encode_param)

            if not success_encode:
                return jsonify({
                    'status': 'error',
                    'message': '最终结果图像编码失败'
                }), 500

            response['final_result_data'] = base64.b64encode(buffer).decode('utf-8')

            # 如果使用了外观增强，返回增强后的图像
            if use_enhancement and enhanced_image is not None:
                if file_extension == 'png':
                    success_encode_enhanced, buffer_enhanced = cv2.imencode('.png', enhanced_image, encode_param)
                else:
                    success_encode_enhanced, buffer_enhanced = cv2.imencode('.jpg', enhanced_image, encode_param)

                if success_encode_enhanced:
                    response['enhanced_image_data'] = base64.b64encode(buffer_enhanced).decode('utf-8')

            # 返回第一次擦除结果
            if first_erase_result is not None:
                if file_extension == 'png':
                    success_encode_first, buffer_first = cv2.imencode('.png', first_erase_result, encode_param)
                else:
                    success_encode_first, buffer_first = cv2.imencode('.jpg', first_erase_result, encode_param)

                if success_encode_first:
                    response['first_erase_data'] = base64.b64encode(buffer_first).decode('utf-8')

            return jsonify(response)
        else:
            return jsonify({
                'status': 'error',
                'message': f'处理失败: {final_result}'
            }), 500

    except Exception as e:
        print(f"处理请求时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'服务器错误: {str(e)}'
        }), 500


@app.route('/process_file', methods=['POST'])
def process_image_file():
    """图像处理接口 - 返回文件下载"""
    try:
        # 检查是否有文件
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '没有上传文件'
            }), 400

        file = request.files['image']

        # 检查文件名
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': '没有选择文件'
            }), 400

        # 检查文件类型
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': f'不支持的文件类型，支持的格式: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
            }), 400

        # 获取处理模式参数
        use_enhancement = request.form.get('use_enhancement', 'true').lower() == 'true'
        additional_erase_count = int(request.form.get('additional_erase_count', '0'))

        # 验证额外擦除次数
        if additional_erase_count < 0 or additional_erase_count > 5:
            return jsonify({
                'status': 'error',
                'message': '额外擦除次数必须在0-5之间'
            }), 400

        # 直接读取文件到内存
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                'status': 'error',
                'message': '无法解析图像文件'
            }), 400

        print(f"开始处理图像，尺寸: {image.shape}, 使用外观增强: {use_enhancement}, 额外擦除次数: {additional_erase_count}")

        # 处理图像
        success, final_result, enhanced_image, first_erase_result, processing_time = processor.process_image(
            image, use_enhancement, additional_erase_count)

        if success:
            # 创建临时文件
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                cv2.imwrite(tmp_file.name, final_result)
                temp_path = tmp_file.name

            # 返回文件并在发送后删除
            def remove_file(response):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"删除临时文件失败: {e}")
                return response

            response = send_file(
                temp_path,
                as_attachment=True,
                download_name=f"processed_{additional_erase_count+1}x_{file.filename}",
                mimetype=f'image/{file_extension}'
            )

            # 注册回调在响应完成后删除文件
            response.call_on_close(lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None)

            return response
        else:
            return jsonify({
                'status': 'error',
                'message': f'处理失败: {final_result}'
            }), 500

    except Exception as e:
        print(f"处理请求时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'服务器错误: {str(e)}'
        }), 500


@app.route('/', methods=['GET'])
def index():
    return '''
    <!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>手写体擦除工具</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            backdrop-filter: blur(10px);
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header .subtitle {
            color: #7f8c8d;
            font-size: 1.1em;
            font-weight: 400;
        }

        .header .icon {
            font-size: 3em;
            margin-bottom: 15px;
            color: #667eea;
        }

        .upload-section {
            border: 3px dashed #bdc3c7;
            border-radius: 15px;
            padding: 50px;
            text-align: center;
            background: linear-gradient(145deg, #f8f9fa, #e9ecef);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .upload-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
            transform: translateX(-100%);
            transition: transform 0.6s;
        }

        .upload-section:hover::before {
            transform: translateX(100%);
        }

        .upload-section.dragover {
            border-color: #667eea;
            background: linear-gradient(145deg, #f0f4ff, #e8f0ff);
            transform: scale(1.02);
        }

        .upload-section .upload-icon {
            font-size: 4em;
            color: #667eea;
            margin-bottom: 20px;
        }

        .upload-section p {
            font-size: 1.2em;
            color: #2c3e50;
            margin-bottom: 25px;
            font-weight: 500;
        }

        #fileInput {
            display: none;
        }

        .upload-btn {
            display: inline-block;
            padding: 15px 35px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: 600;
            transition: all 0.3s ease;
            border: none;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }

        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }

        .file-info {
            margin-top: 25px;
            padding: 20px;
            background: linear-gradient(145deg, #e8f4f8, #d1ecf1);
            border-radius: 12px;
            display: none;
            border-left: 4px solid #667eea;
        }

        .process-options {
            margin: 40px 0;
        }

        .option-group {
            margin: 25px 0;
            padding: 25px;
            background: linear-gradient(145deg, #ffffff, #f8f9fa);
            border-radius: 15px;
            border: 1px solid #e9ecef;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }

        .option-group h3 {
            margin-bottom: 15px;
            color: #2c3e50;
            font-size: 1.2em;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .option-group h3::before {
            content: '⚙️';
            font-size: 1.2em;
        }

        .process-options label {
            margin: 10px 20px 10px 0;
            font-size: 1em;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            padding: 8px 15px;
            border-radius: 20px;
            transition: all 0.3s ease;
            background: rgba(102, 126, 234, 0.05);
        }

        .process-options label:hover {
            background: rgba(102, 126, 234, 0.1);
            transform: translateY(-1px);
        }

        .process-options input[type="radio"] {
            margin-right: 10px;
            accent-color: #667eea;
        }

        .erase-count-group {
            margin: 20px 0;
            text-align: center;
        }

        .erase-count-group > label {
            display: block;
            margin-bottom: 15px;
            font-weight: 600;
            color: #2c3e50;
            font-size: 1.1em;
        }

        .slider-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin: 20px 0;
        }

        .erase-count-group input[type="range"] {
            width: 300px;
            height: 8px;
            border-radius: 5px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
        }

        .erase-count-group input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(45deg, #667eea, #764ba2);
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        .count-display {
            font-size: 1.5em;
            color: #667eea;
            font-weight: bold;
            background: linear-gradient(145deg, #f0f4ff, #e8f0ff);
            padding: 5px 15px;
            border-radius: 20px;
            min-width: 40px;
            text-align: center;
        }

        .erase-info {
            margin-top: 15px;
            padding: 15px;
            background: linear-gradient(145deg, #e3f2fd, #bbdefb);
            border-radius: 10px;
            color: #1565c0;
            font-size: 0.9em;
            border-left: 4px solid #2196f3;
        }

        .process-btn {
            display: none;
            margin: 30px auto;
            padding: 18px 50px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 30px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }

        .process-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
        }

        .process-btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .loading {
            display: none;
            text-align: center;
            margin: 30px 0;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        .loading p {
            color: #667eea;
            font-size: 1.1em;
            font-weight: 500;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .result-section {
            display: none;
            margin-top: 40px;
            padding: 30px;
            background: linear-gradient(145deg, #f8f9fa, #ffffff);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .result-section h2 {
            text-align: center;
            margin-bottom: 30px;
            color: #2c3e50;
            font-size: 1.8em;
            font-weight: 600;
        }

        .result-images {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }

        .image-container {
            position: relative;
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }

        .image-container:hover {
            transform: translateY(-5px);
        }

        .image-container h3 {
            text-align: center;
            padding: 15px;
            margin: 0;
            color: #2c3e50;
            font-size: 1em;
            font-weight: 600;
            background: linear-gradient(145deg, #f8f9fa, #e9ecef);
            border-bottom: 1px solid #dee2e6;
        }

        .image-container img {
            width: 100%;
            height: auto;
            display: block;
        }

        .download-btn {
            position: absolute;
            top: 50px;
            right: 15px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 8px 15px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 500;
            transition: all 0.3s ease;
            z-index: 10;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }

        .download-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }

        .error-message {
            display: none;
            padding: 20px;
            background: linear-gradient(145deg, #fee, #fdd);
            border: 1px solid #fcc;
            border-radius: 12px;
            color: #c33;
            margin: 20px 0;
            border-left: 4px solid #e74c3c;
        }

        .success-message {
            display: none;
            padding: 20px;
            background: linear-gradient(145deg, #efe, #dfd);
            border: 1px solid #cfc;
            border-radius: 12px;
            color: #27ae60;
            margin: 20px 0;
            border-left: 4px solid #27ae60;
        }

        .stats {
            margin-top: 20px;
            padding: 20px;
            background: linear-gradient(145deg, #e8f4f8, #d1ecf1);
            border-radius: 12px;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            line-height: 1.6;
            border-left: 4px solid #667eea;
        }

        .enhancement-notice {
            background: linear-gradient(145deg, #fff3cd, #ffeaa7);
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            text-align: center;
            font-size: 0.9em;
            border-left: 4px solid #f39c12;
        }



        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }

            .header h1 {
                font-size: 2em;
            }

            .upload-section {
                padding: 30px 20px;
            }

            .result-images {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="icon">✏️</div>
            <h1>手写体擦除工具</h1>
            <p class="subtitle">智能识别并擦除图片中的手写内容，让您的文档更加专业</p>
        </div>



        <div class="upload-section" id="uploadSection">
            <div class="upload-icon">📁</div>
            <p>将图片拖拽到此处或点击上传</p>
            <label for="fileInput" class="upload-btn">选择图片文件</label>
            <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.bmp">
            <div class="file-info" id="fileInfo"></div>
        </div>

        <div class="process-options">
            <div class="option-group">
                <h3>处理模式</h3>
                <label>
                    <input type="radio" name="enhancementMode" value="true" checked>
                    智能增强 + 手写擦除
                </label>
                <label>
                    <input type="radio" name="enhancementMode" value="false">
                    仅手写擦除
                </label>
            </div>

            <div class="option-group">
                <h3>擦除设置</h3>
                <div class="erase-count-group">
                    <label>额外擦除次数：</label>
                    <div class="slider-container">
                        <span>0</span>
                        <input type="range" id="eraseCountSlider" min="0" max="5" value="0" step="1">
                        <span>5</span>
                        <div class="count-display" id="eraseCountDisplay">0</div>
                    </div>
                    <div class="erase-info">
                        💡 总擦除次数 = 1 + 额外擦除次数。增加擦除次数可以获得更彻底的清理效果
                    </div>
                </div>
            </div>


        </div>

        <button class="process-btn" id="processBtn">🚀 开始处理</button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>正在处理中，请稍候...</p>
        </div>

        <div class="error-message" id="errorMessage"></div>
        <div class="success-message" id="successMessage"></div>

        <div class="result-section" id="resultSection">
            <h2>🎉 处理结果</h2>
            <div class="stats" id="stats"></div>
            <div id="enhancementNotice" class="enhancement-notice" style="display: none;">
                📋 注意：当前模式未使用智能增强，直接进行手写擦除处理
            </div>
            <div class="result-images" id="resultImages">
                <div class="image-container">
                    <h3>📷 原始图片</h3>
                    <img id="originalImage" alt="原始图片">
                    <button class="download-btn" onclick="downloadImage('originalImage', 'original')">下载</button>
                </div>
                <div class="image-container" id="enhancedContainer" style="display: none;">
                    <h3>✨ 智能增强后</h3>
                    <img id="enhancedImage" alt="智能增强后图片">
                    <button class="download-btn" onclick="downloadImage('enhancedImage', 'enhanced')">下载</button>
                </div>
                <div class="image-container" id="firstEraseContainer">
                    <h3>🎯 第一次擦除</h3>
                    <img id="firstEraseImage" alt="第一次擦除结果">
                    <button class="download-btn" onclick="downloadImage('firstEraseImage', 'first_erase')">下载</button>
                </div>
                <div class="image-container" id="finalContainer">
                    <h3 id="finalTitle">🏆 最终结果</h3>
                    <img id="finalImage" alt="最终处理结果">
                    <button class="download-btn" onclick="downloadImage('finalImage', 'final')">下载</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE_URL = 'http://localhost:5003';

        const uploadSection = document.getElementById('uploadSection');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const processBtn = document.getElementById('processBtn');
        const loading = document.getElementById('loading');
        const errorMessage = document.getElementById('errorMessage');
        const successMessage = document.getElementById('successMessage');
        const resultSection = document.getElementById('resultSection');
        const originalImage = document.getElementById('originalImage');
        const enhancedImage = document.getElementById('enhancedImage');
        const firstEraseImage = document.getElementById('firstEraseImage');
        const finalImage = document.getElementById('finalImage');
        const enhancedContainer = document.getElementById('enhancedContainer');
        const firstEraseContainer = document.getElementById('firstEraseContainer');
        const finalContainer = document.getElementById('finalContainer');
        const enhancementNotice = document.getElementById('enhancementNotice');
        const stats = document.getElementById('stats');
        const eraseCountSlider = document.getElementById('eraseCountSlider');
        const eraseCountDisplay = document.getElementById('eraseCountDisplay');
        const finalTitle = document.getElementById('finalTitle');

        let selectedFile = null;

        // 擦除次数滑块事件
        eraseCountSlider.addEventListener('input', (e) => {
            const count = parseInt(e.target.value);
            eraseCountDisplay.textContent = count;

            if (count === 0) {
                finalTitle.textContent = '🎯 第一次擦除';
                finalContainer.style.display = 'none';
            } else {
                finalTitle.textContent = `🏆 最终结果 (${count + 1}次擦除)`;
                finalContainer.style.display = 'block';
            }
        });

        // 文件选择
        fileInput.addEventListener('change', handleFileSelect);

        // 拖拽上传
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });

        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });

        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        function handleFileSelect(e) {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        }

        function handleFile(file) {
            const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp'];

            if (!allowedTypes.includes(file.type)) {
                showError('❌ 请选择支持的图片格式（PNG, JPG, JPEG, BMP）');
                return;
            }

            if (file.size > 16 * 1024 * 1024) {
                showError('❌ 文件大小不能超过16MB');
                return;
            }

            selectedFile = file;
            fileInfo.style.display = 'block';
            fileInfo.innerHTML = `
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div style="font-size: 2em;">📄</div>
                    <div>
                        <div><strong>文件名:</strong> ${file.name}</div>
                        <div><strong>大小:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB</div>
                        <div><strong>类型:</strong> ${file.type}</div>
                    </div>
                </div>
            `;

            // 显示原始图片预览
            const reader = new FileReader();
            reader.onload = (e) => {
                originalImage.src = e.target.result;
            };
            reader.readAsDataURL(file);

            processBtn.style.display = 'block';
            hideMessages();
        }

        // 处理按钮点击
        processBtn.addEventListener('click', async () => {
            if (!selectedFile) {
                showError('❌ 请先选择一个图片文件');
                return;
            }

            const enhancementMode = document.querySelector('input[name="enhancementMode"]:checked').value;
            const additionalEraseCount = parseInt(eraseCountSlider.value);

            processBtn.disabled = true;
            loading.style.display = 'block';
            hideMessages();
            resultSection.style.display = 'none';

            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('use_enhancement', enhancementMode);
            formData.append('additional_erase_count', additionalEraseCount.toString());

            try {
                const startTime = Date.now();

                // Base64 模式
                const response = await fetch(`${API_BASE_URL}/process`, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                    if (data.status === 'success') {
                        const endTime = Date.now();
                        const totalTime = ((endTime - startTime) / 1000).toFixed(2);

                        // 显示处理后的图片
                        if (data.final_result_data) {
                            finalImage.src = `data:image/${data.format};base64,${data.final_result_data}`;
                        }

                        // 处理外观增强图片
                        if (data.use_enhancement && data.enhanced_image_data) {
                            enhancedImage.src = `data:image/${data.format};base64,${data.enhanced_image_data}`;
                            enhancedContainer.style.display = 'block';
                            enhancementNotice.style.display = 'none';
                        } else {
                            enhancedContainer.style.display = 'none';
                            enhancementNotice.style.display = data.use_enhancement ? 'none' : 'block';
                        }

                        // 显示第一次擦除结果
                        if (data.first_erase_data) {
                            firstEraseImage.src = `data:image/${data.format};base64,${data.first_erase_data}`;
                        }

                        // 根据擦除次数调整显示
                        if (data.additional_erase_count === 0) {
                            finalContainer.style.display = 'none';
                            finalTitle.textContent = '🎯 第一次擦除';
                        } else {
                            finalContainer.style.display = 'block';
                            finalTitle.textContent = `🏆 最终结果 (${data.total_erase_count}次擦除)`;
                        }

                        // 显示统计信息
                        stats.innerHTML = `
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                                <div><strong>🔧 处理模式:</strong> ${data.use_enhancement ? '智能增强 + 手写擦除' : '仅手写擦除'}</div>
                                <div><strong>🎯 总擦除次数:</strong> ${data.total_erase_count}次</div>
                                <div><strong>➕ 额外擦除:</strong> ${data.additional_erase_count}次</div>
                                <div><strong>⏱️ 处理时间:</strong> ${data.processing_time}</div>
                                <div><strong>🌐 总耗时:</strong> ${totalTime}秒</div>
                                <div><strong>📷 图片格式:</strong> ${data.format.toUpperCase()}</div>
                            </div>
                        `;

                        resultSection.style.display = 'block';
                        showSuccess('✅ 图片处理成功！手写内容已被智能擦除');
                    } else {
                        showError('❌ ' + (data.message || '处理失败'));
                    }
            } catch (error) {
                showError('❌ 网络错误: ' + error.message);
            } finally {
                processBtn.disabled = false;
                loading.style.display = 'none';
            }
        });

        // 下载图片功能
        function downloadImage(imgId, prefix) {
            const img = document.getElementById(imgId);
            if (!img.src) {
                showError('❌ 图片未加载');
                return;
            }

            // 创建canvas来转换图片
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            const image = new Image();
            image.crossOrigin = 'anonymous';
            image.onload = function() {
                canvas.width = image.width;
                canvas.height = image.height;
                ctx.drawImage(image, 0, 0);

                canvas.toBlob(function(blob) {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${prefix}_${selectedFile ? selectedFile.name : 'image.jpg'}`;
                    document.body.appendChild(a);
                    a.click();
                    URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                }, 'image/jpeg', 0.9);
            };
            image.src = img.src;
        }

        function showError(message) {
            errorMessage.innerHTML = message;
            errorMessage.style.display = 'block';
            successMessage.style.display = 'none';
        }

        function showSuccess(message) {
            successMessage.innerHTML = message;
            successMessage.style.display = 'block';
            errorMessage.style.display = 'none';
        }

        function hideMessages() {
            errorMessage.style.display = 'none';
            successMessage.style.display = 'none';
        }
    </script>
</body>
</html>
    '''


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'status': 'error',
        'message': '文件过大，最大支持16MB'
    }), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'status': 'error',
        'message': '服务器内部错误'
    }), 500


if __name__ == '__main__':
    # 初始化处理器
    try:
        init_processor(torch_device_id=0, paddle_device_id=0)
        print("Flask应用启动中...")
        app.run(host='127.0.0.1', port=5003, debug=False)
    except Exception as e:
        print(f"应用启动失败: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)