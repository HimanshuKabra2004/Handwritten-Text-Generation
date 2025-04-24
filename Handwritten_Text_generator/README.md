# Handwritten Text Generator

This project generates handwritten-style images from text input using a trained deep learning model.

## Features
- Load IAM handwritten dataset
- Train a CNN-based image generation model
- Generate handwritten text images
- Bonus: Apply style transfer
- Bonus: Match user input to best font style

## Usage
```bash
python main.py
```

## Folder Structure
```
handwritten-text-generator/
├── data/
│   └── iam/
├── notebooks/
├── models/
├── outputs/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   ├── style_transfer.py
│   ├── user_input_matcher.py
├── README.md
├── requirements.txt
└── main.py
