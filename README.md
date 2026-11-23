# Crop Disease Detection Using Deep Learning

## Overview

This project focuses on crop disease classification using Deep Learning and Computer Vision. It uses a MobileNetV3-based Convolutional Neural Network (CNN) to identify diseases from plant leaf images.

The model classifies images into 15 categories across four crops: Maize, Potato, Rice, and Tomato.

The model was trained using PyTorch and exported to ONNX format to support efficient inference and integration into external applications.

## Supported Crops and Diseases

| Crop | Classes |
|---|---|
| Maize | Common Rust, Gray Leaf Spot, Healthy, Northern Leaf Blight |
| Potato | Early Blight, Healthy, Late Blight |
| Rice | Brown Spot, Healthy, Leaf Blast |
| Tomato | Early Blight, Healthy, Late Blight, Leaf Mold, Septoria Leaf Spot |

## Model Architecture

**MobileNetV3**

MobileNetV3 is a lightweight CNN architecture designed for efficient image classification. It uses computationally efficient operations and is suitable for applications where inference speed and computational resources are important.

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | MobileNetV3 |
| Task | Multi-class Image Classification |
| Number of Classes | 15 |
| Input Size | 224 × 224 |
| Input Channels | 3 (RGB) |
| Loss Function | Cross Entropy Loss |
| Framework | PyTorch |
| Deployment Format | ONNX |

## Dataset and Preprocessing

The project uses plant leaf images from the PlantVillage dataset and rice disease images.

The preprocessing pipeline includes:

- Image resizing and cropping to 224 × 224 pixels.
- Random horizontal and vertical flipping.
- Color jittering for brightness, contrast, saturation, and hue.
- Random Gaussian blur.
- Tensor conversion.
- Image normalization using ImageNet mean and standard deviation.

During inference, random augmentations are disabled. Images are resized to 224 × 224 and normalized using the same ImageNet statistics.

## Model Performance

The model was evaluated on a separate test dataset containing 2,401 images.

| Metric | Score |
|---|---:|
| Accuracy | 96.7% |
| Macro Precision | 95.8% |
| Macro Recall | 95.6% |
| Macro F1-Score | 95.7% |
| Weighted F1-Score | 96.6% |

### ONNX Validation

The trained PyTorch model was exported to ONNX format and validated using ONNX Runtime.

| Validation Parameter | Result |
|---|---|
| PyTorch Output Shape | (1, 15) |
| ONNX Output Shape | (1, 15) |
| PyTorch Predicted Class | 8 |
| ONNX Predicted Class | 8 |
| Maximum Absolute Difference | 2.3841858e-06 |

The ONNX model was additionally tested on 100 randomly selected test images and achieved 97% accuracy.

## Technologies Used

- Python
- PyTorch
- Torchvision
- MobileNetV3
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Pillow
- Matplotlib
- ONNX
- ONNX Runtime
- Kaggle

## Project Structure

```text
Crop-Disease-Detection/
│
├── notebooks/
│   └── crop_disease_training.ipynb
│
├── model/
│   ├── crop_disease_model.onnx
│   └── labels.json
│
├── inference/
│   └── inference.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/mohantripathi90706-svg/Crop-Disease-Detection.git
```

Navigate to the project directory:

```bash
cd Crop-Disease-Detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Model Inference

The project includes an ONNX-based inference pipeline for predicting crop diseases from leaf images.

### Example

```python
from inference.inference import CropDiseasePredictor

predictor = CropDiseasePredictor(
    model_path="model/crop_disease_model.onnx",
    labels_path="model/labels.json"
)

predictions = predictor.predict("test_image.jpg")

for result in predictions:
    print(result)
```

### Sample Output

```text
[
    {
        'disease': 'tomato_early_blight',
        'confidence': 86.11
    }
]
```

The inference pipeline performs the following steps:

1. Loads the input leaf image.
2. Converts the image to RGB format.
3. Resizes the image to 224 × 224 pixels.
4. Applies image normalization.
5. Passes the processed image to the ONNX model.
6. Generates predictions across 15 disease classes.
7. Returns the top predicted classes with their confidence scores.

## Training and Evaluation

The training notebook contains the model development workflow:

- Dataset loading and exploration.
- Data preprocessing and augmentation.
- MobileNetV3 model training.
- Model evaluation.
- Classification report generation.
- Confusion matrix visualization.
- Saving trained model weights.
- Exporting the model to ONNX.
- ONNX model validation.
- PyTorch and ONNX output comparison.

The training notebook was developed in Kaggle. Dataset paths may need to be modified when running it in another environment.

## Contributors

**Mohan Tripathi**

B.Tech, Computer Science and Engineering  
Shri Mata Vaishno Devi University (SMVDU)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
