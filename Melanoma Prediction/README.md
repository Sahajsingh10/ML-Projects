# Skin Melanoma Prediction

A **PyTorch convolutional neural network (CNN)** for binary classification of dermoscopic skin-lesion images as **benign** or **malignant**.

The project covers data loading, image preprocessing and augmentation, CNN training, evaluation on a held-out test set, model-weight persistence, and inference on unseen images.

## Dataset

The notebook uses the Kaggle **Melanoma Skin Cancer Dataset of 10,000 Images**.

The loaded dataset contains:

- **9,605 training images**
- **1,000 test images**
- Two classes: **benign** and **malignant**

## Model Architecture

The CNN is implemented from scratch in PyTorch and includes:

```text
Input image (3 x 200 x 200)
        |
        v
Conv2D: 3 -> 32
BatchNorm
ReLU
MaxPool
        |
        v
Conv2D: 32 -> 64
BatchNorm
ReLU
MaxPool
        |
        v
Flatten
        |
        v
Fully Connected: 160000 -> 128
ReLU
        |
        v
Fully Connected: 128 -> 2
        |
        v
Benign / Malignant
```

## Image Preprocessing and Augmentation

Images are resized to **200 x 200** and processed with:

- Random horizontal flips
- Random vertical flips
- Random rotations up to 10 degrees
- Tensor conversion
- Normalization

These augmentations introduce variation during training and help the model generalize beyond the original training images.

## Training

The notebook trains the CNN with:

- **PyTorch**
- Cross-entropy loss
- SGD optimizer
- Learning rate: `0.001`
- Momentum: `0.9`
- Batch size: `20`
- **10 epochs**
- GPU acceleration when available

Training accuracy increased from approximately **85.7% in epoch 1** to **90.9% by epoch 10**.

## Evaluation

On the held-out test set, the notebook reports:

- **Test accuracy: 90.9%**
- Test loss: approximately **0.228**

The project also saves the model weights and demonstrates inference on external example images.

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow
- Kaggle
- Google Colab

## Running the Project

1. Open `Melanoma_Prediction.ipynb` in Google Colab.
2. Enable a GPU runtime.
3. Configure Kaggle credentials.
4. Download the dataset using the notebook's Kaggle command.
5. Run the preprocessing, training, and evaluation cells in order.

The notebook includes Google Drive paths for saved model weights and sample inference images, so those paths may need to be changed for your environment.

## Repository Contents

- `Melanoma_Prediction.ipynb` — complete training and inference workflow
- `Melanoma Prediction.pdf` — exported project notebook/report

## Disclaimer

This project is for educational and machine-learning experimentation purposes only. It is **not a medical diagnostic system** and should not be used for clinical decision-making.
