[简体中文](./Readme.md) | English

# A Simple Handwriting Removal Tool

<div align="center">
    <img src="images\HandWritingRemove_logo.svg" alt="logo" style="zoom:800%;" />
</div>

![HandWritingRemove](https://img.shields.io/badge/HandWritingRemove-Handwriting%20Removal%20Tool-red) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![Flask](https://img.shields.io/badge/Flask-2.0+-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-blue) ![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-2.0+-yellow)

**Deep Learning-based Intelligent Handwriting Removal System**

## 📖 Project Introduction

A Simple Handwriting Removal Tool is an intelligent handwritten content removal tool based on deep learning technology, capable of automatically identifying and erasing handwritten text in images, making documents look cleaner and more professional. The system features the following core capabilities:

### Environment Installation

```bash
# Install PyTorch (CUDA 11.8)
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
# Install other dependencies
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Model File Preparation

Store the following model files in the project directory:

- **Appearance Enhancement Model Weights**:
  - `./ckpts/gcdr_net/gcnet/checkpoint.pkl` - GCNet shadow removal model
  - `./ckpts/gcdr_net/drnet/checkpoint.pkl` - DRNet document restoration model
- **Handwriting Removal Model Weights**:
  - `./ckpts/STE_idr_best.pdparams` - STRAIDR handwriting removal model

The above model weight files have been uploaded to [Baidu Netdisk](https://pan.baidu.com/s/1H74KpxvJXNYP_RJwFfSvhw), with extraction code `3p4y`

### Start Service

```bash
# Set GPU device (optional)
export CUDA_VISIBLE_DEVICES=1
# Start Flask service
python flask_hwRemove.py
```

After successful startup, visit `http://localhost:5003` to use the web interface for handwriting removal.

## 📡 API Interface

### Handwriting Removal Interface

**Endpoint**: `POST /process`

**Request Parameters**:

| Parameter                | Type    | Required | Description                                    |
| ------------------------ | ------- | -------- | ---------------------------------------------- |
| image                    | file    | Yes      | Image file (supports PNG/JPG/JPEG/BMP, ≤16MB) |
| use_enhancement          | boolean | No       | Whether to use appearance enhancement (default: true) |
| additional_erase_count   | integer | No       | Additional erase count, 0-10 (default: 0)     |

**Request Examples**:

```bash
# Smart enhancement + single erase
curl -X POST http://localhost:5003/process \
  -F "image=@handwritten_document.jpg" \
  -F "use_enhancement=true" \
  -F "additional_erase_count=0"

# Handwriting removal only + multiple iterations
curl -X POST http://localhost:5003/process \
  -F "image=@handwritten_document.jpg" \
  -F "use_enhancement=false" \
  -F "additional_erase_count=2"
```

**Success Response Example**:

```json
{
  "status": "success",
  "format": "jpg",
  "total_erase_count": 1,
  "final_result_data": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**Response Field Description**:
- `status`: Processing status (success/error)
- `format`: Image format
- `total_erase_count`: Total erase count
- `final_result_data`: Final processing result (Base64 encoded)

## 📁 Project Structure

```
HandWritingRemove/
├── flask_hwRemove.py          # Flask main application
├── models/                    # Deep learning model definitions
│   ├── unext.py              # UNext appearance enhancement model
│   └── sa_aidr.py            # STRAIDR handwriting removal model
├── utils/                     # Utility functions
│   └── utils_gcdrnet.py      # GCDR network related tools
├── ckpts/                     # Model weight file directory
│   ├── gcdr_net/             # Appearance enhancement model weights
│   └── STE_idr_best.pdparams # Handwriting removal model weights
├── requirements.txt           # Python dependency list
├── images/                    # Sample images and resources
└── README.md                 # Project documentation
```

## 🎨 Removal Effect Showcase

### Web Interface
<div align="center">
    <img src="images\web_show.png" alt="Web interface showcase"/>
</div>

### Demo
<div align="center">
    <img src="images\demo.png" alt="Processing effect comparison"/>
</div>