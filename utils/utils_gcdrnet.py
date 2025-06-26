from collections import OrderedDict
import os
import numpy as np
import torch
import random
import torchvision
import torch.nn.functional as F
import os
import cv2
from tqdm import tqdm


def dict2string(loss_dict):
    loss_string = ''
    for key, value in loss_dict.items():
        loss_string += key + ' {:.4f}, '.format(value)
    return loss_string[:-2]


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])


def torch2cvimg(tensor, min=0, max=1):
    '''
    input:
        tensor -> torch.tensor BxCxHxW C can be 1,3
    return
        im -> ndarray uint8 HxWxC
    '''
    im_list = []
    for i in range(tensor.shape[0]):
        im = tensor.detach().cpu().data.numpy()[i]
        im = im.transpose(1, 2, 0)
        im = np.clip(im, min, max)
        im = ((im - min) / (max - min) * 255).astype(np.uint8)
        im_list.append(im)
    return im_list


def cvimg2torch(img, min=0, max=1):
    '''
    input:
        im -> ndarray uint8 HxWxC
    return
        tensor -> torch.tensor BxCxHxW
    '''
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)  # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    return img


def setup_seed(seed):
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed) #cpu
    # torch.cuda.manual_seed_all(seed)  #并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    # torch.backends.cudnn.benchmark = False   #训练集变化不大时使训练加速


def stride_integral(img, stride=32):
    h, w = img.shape[:2]

    if (h % stride) != 0:
        padding_h = stride - (h % stride)
        img = cv2.copyMakeBorder(img, padding_h, 0, 0, 0, borderType=cv2.BORDER_REPLICATE)
    else:
        padding_h = 0

    if (w % stride) != 0:
        padding_w = stride - (w % stride)
        img = cv2.copyMakeBorder(img, 0, 0, padding_w, 0, borderType=cv2.BORDER_REPLICATE)
    else:
        padding_w = 0

    return img, padding_h, padding_w


def test_model1_model2(model1, model2, path_list, in_folder, sav_folder):
    for im_path in tqdm(path_list):
        in_name = im_path.split('_')[-1].split('.')[0]

        im_org = cv2.imread(im_path)
        im_org, padding_h, padding_w = stride_integral(im_org)
        h, w = im_org.shape[:2]
        im = cv2.resize(im_org, (512, 512))
        im = im_org
        with torch.no_grad():
            im = torch.from_numpy(im.transpose(2, 0, 1) / 255).unsqueeze(0)
            # im = im.float().cuda()
            im = im.float().cpu()
            im_org = torch.from_numpy(im_org.transpose(2, 0, 1) / 255).unsqueeze(0)
            # im_org = im_org.float().cuda()
            im_org = im_org.float().cpu()

            shadow = model1(im)
            shadow = F.interpolate(shadow, (h, w))

            model1_im = torch.clamp(im_org / shadow, 0, 1)
            pred, _, _, _ = model2(torch.cat((im_org, model1_im), 1))

            shadow = shadow[0].permute(1, 2, 0).data.cpu().numpy()
            shadow = (shadow * 255).astype(np.uint8)
            shadow = shadow[padding_h:, padding_w:]

            model1_im = model1_im[0].permute(1, 2, 0).data.cpu().numpy()
            model1_im = (model1_im * 255).astype(np.uint8)
            model1_im = model1_im[padding_h:, padding_w:]

            pred = pred[0].permute(1, 2, 0).data.cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            pred = pred[padding_h:, padding_w:]

        # cv2.imwrite(im_path.replace(in_folder,sav_folder),pred)
        output_filename = os.path.basename(im_path)  # 提取纯文件名（如 test.jpg）
        output_path = os.path.join(sav_folder, output_filename)  # 拼接输出路径
        cv2.imwrite(output_path, pred)