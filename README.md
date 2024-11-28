# Vision Transformer (ViT) Classifier for CIFAR-10

This repository implements a Vision Transformer (ViT) model to classify images in the CIFAR-10 dataset. The ViT architecture utilizes a Transformer-based approach for image recognition by dividing images into smaller patches and encoding them with self-attention mechanisms.

---

## Project Overview

Transformers have revolutionized natural language processing and are now being applied to computer vision tasks. This project demonstrates the implementation of a ViT model on the CIFAR-10 dataset, which consists of 60,000 images (32x32) belonging to 10 different classes. The ViT model leverages patch-based feature extraction and multi-head self-attention for image classification.

---

## Key Features

- **Data Augmentation**: Incorporates image resizing, flipping, rotation, and zooming to enhance model generalization.
- **Patch Extraction**: Images are divided into smaller patches to feed into the Transformer architecture.
- **Multi-Head Self-Attention**: Captures spatial dependencies across patches.
- **Custom MLP Heads**: Multi-Layer Perceptrons (MLP) layers for feature processing and classification.
- **Learning Rate Scheduling**: Utilizes the AdamW optimizer for efficient learning.
- **Model Checkpointing**: Saves the best-performing model during training.

---

## Dataset

- **CIFAR-10**: The dataset contains 60,000 images (50,000 for training and 10,000 for testing) in 10 classes:
  - `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

---

## Model Architecture

1. **Data Augmentation Layer**: Preprocesses and augments images.
2. **Patch Extraction**: Converts images into non-overlapping patches.
3. **Patch Encoding**: Embeds patches into a dense representation and adds positional encoding.
4. **Transformer Blocks**:
   - Layer Normalization
   - Multi-Head Self-Attention
   - MLP with skip connections
5. **Classification Head**:
   - Dense layers followed by a final softmax layer for classification.

---

## Installation and Requirements

### Dependencies

- Python 3.8+
- TensorFlow 2.6+ (with Keras)
- TensorFlow Addons
- NumPy
- Matplotlib

Install the required dependencies using:

```bash
pip install tensorflow tensorflow-addons numpy matplotlib
```

## Making Prediction

- Test the model on a sample image from the test set:
``` bash
index = 10  # Index of the image in the test set
plt.imshow(x_test[index])
prediction = img_predict(x_test[index], vit_classifier)
print("Prediction:", prediction)
```

## Evaluation

Evaluate the model's performance on the test dataset:

```python
# Evaluate the model's accuracy and top-5 accuracy
_, accuracy, top_5_accuracy = vit_classifier.evaluate(x_test, y_test)

# Print the evaluation results
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Top-5 Accuracy: {top_5_accuracy * 100:.2f}%")
```

### Expected Outputs:

- **Test Accuracy**: ~85-90%
- **Test Top-5 Accuracy** ~98%

## Refrences

- **CIFAR-10 Dataset**: https://www.cs.toronto.edu/~kriz/cifar.html


