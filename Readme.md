简体中文 | [English](./Readme_en.md)

# 一个普通的擦除手写体

<div align="center">
    <img src="images\HandWritingRemove_logo.svg" alt="logo" style="zoom:800%;" />
</div>

![HandWritingRemove](https://img.shields.io/badge/HandWritingRemove-手写体擦除工具-red) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![Flask](https://img.shields.io/badge/Flask-2.0+-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-blue) ![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-2.0+-yellow)

**基于深度学习的智能手写体擦除系统**

## 📖 项目简介

一个普通的擦除手写体是一个基于深度学习技术的智能手写内容移除工具，能够自动识别并擦除图片中的手写文字，让文档看起来更加整洁专业。系统具备以下核心能力：

### 环境安装

```bash
# 安装PyTorch (CUDA 11.8)
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
# 安装其他依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 模型文件准备

将以下模型文件存储于项目目录下：

- **外观增强模型权重**:
  - `./ckpts/gcdr_net/gcnet/checkpoint.pkl` - GCNet阴影消除模型
  - `./ckpts/gcdr_net/drnet/checkpoint.pkl` - DRNet文档修复模型
- **手写擦除模型权重**:
  - `./ckpts/STE_idr_best.pdparams` - STRAIDR手写擦除模型

上述模型权重文件已上传至[百度网盘](https://pan.baidu.com/s/1H74KpxvJXNYP_RJwFfSvhw)，提取码为`3p4y` 

### 启动服务

```bash
# 设置GPU设备（可选）
export CUDA_VISIBLE_DEVICES=1
# 启动Flask服务
python flask_hwRemove.py
```

启动成功后访问 `http://localhost:5003` 使用Web界面进行手写体擦除。

## 📡 API接口

### 手写体擦除接口

**端点**: `POST /process`

**请求参数**:

| 参数                     | 类型    | 必填 | 说明                                    |
| ------------------------ | ------- | ---- | --------------------------------------- |
| image                    | file    | 是   | 图片文件（支持 PNG/JPG/JPEG/BMP，≤16MB） |
| use_enhancement          | boolean | 否   | 是否使用外观增强（默认：true）              |
| additional_erase_count   | integer | 否   | 额外擦除次数，0-10（默认：0）              |

**请求示例**:

```bash
# 智能增强 + 单次擦除
curl -X POST http://localhost:5003/process \
  -F "image=@handwritten_document.jpg" \
  -F "use_enhancement=true" \
  -F "additional_erase_count=0"

# 仅手写擦除 + 多次迭代
curl -X POST http://localhost:5003/process \
  -F "image=@handwritten_document.jpg" \
  -F "use_enhancement=false" \
  -F "additional_erase_count=2"
```

**成功响应示例**:

```json
{
  "status": "success",
  "format": "jpg",
  "total_erase_count": 1,
  "final_result_data": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**响应字段说明**:
- `status`: 处理状态（success/error）
- `format`: 图片格式
- `total_erase_count`: 总擦除次数
- `final_result_data`: 最终处理结果（Base64编码）

## 📁 项目结构

```
HandWritingRemove/
├── flask_hwRemove.py          # Flask主应用程序
├── models/                    # 深度学习模型定义
│   ├── unext.py              # UNext外观增强模型
│   └── sa_aidr.py            # STRAIDR手写擦除模型
├── utils/                     # 工具函数
│   └── utils_gcdrnet.py      # GCDR网络相关工具
├── ckpts/                     # 模型权重文件目录
│   ├── gcdr_net/             # 外观增强模型权重
│   └── STE_idr_best.pdparams # 手写擦除模型权重
├── requirements.txt           # Python依赖列表
├── images/                    # 示例图片和资源
└── README.md                 # 项目说明文档
```

## 🎨 擦除效果展示

### Web界面
<div align="center">
    <img src="images\web_show.png" alt="web界面展示"/>
</div>

### demo
<div align="center">
    <img src="images\demo.png" alt="处理效果对比"/>
</div>
