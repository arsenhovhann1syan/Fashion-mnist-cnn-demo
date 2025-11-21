## FashionMNIST Image Classification (PyTorch CNN)

---

###  Project Overview

This project implements an image classification solution for the **Fashion-MNIST** dataset using **PyTorch** and a **Convolutional Neural Network (CNN)** architecture.

Fashion-MNIST consists of 70,000 grayscale images (28x28 pixels) across **10 categories** of clothing, serving as a more challenging alternative to the classic MNIST dataset. The model achieved a competitive test accuracy of **92.69%** over 20 epochs.

### Architecture and Parameters

#### Model: `FashionCNN` (CNN)
The model utilizes a simple yet effective CNN structure for feature extraction and classification.

* **Convolutional Layers:** `Conv2d(1, 32)` -> `Conv2d(32, 64)` followed by `ReLU` and `MaxPool2d(2, 2)`.
* **Fully Connected Layers:** Two linear layers with a dropout layer (`Dropout(0.5)`) for regularization.
* **Final Output:** 10 classes.

#### Class Labels
The 10 classes in the Fashion-MNIST dataset are mapped as follows:

| Code | Description |
| :--- | :--- |
| **0** | T-shirt/top |
| **1** | Trouser |
| **2** | Pullover |
| **3** | Dress |
| **4** | Coat |
| **5** | Sandal |
| **6** | Shirt |
| **7** | Sneaker |
| **8** | Bag |
| **9** | Ankle boot |

#### Hyperparameters

| Parameter | Value |
| :--- | :--- |
| **Epochs** | 20 |
| **Batch Size** | 128 |
| **Learning Rate** | $1e-3$ |
| **Optimizer** | Adam |
| **Loss Function** | CrossEntropyLoss |

### Results and Evaluation

#### Final Model Performance Summary

| Metric | Value | Epoch |
| :--- | :--- | :--- |
| **Best Test Accuracy** | **93.04%** | 17 |
| **Lowest Test Loss** | **0.2126** | 9 |
| Final Test Accuracy | 92.69% | 20 |
| Final Train Loss | 0.0952 | 20 |
| Final Test Loss | 0.2549 | 20 |

#### Per-Class Performance

The per-class analysis highlights the model's strengths and weaknesses.

| Digit | Description | Accuracy (%) | Errors |
| :--- | :--- | :--- | :--- |
| **0** | T-shirt/top | 88.50 | 115 |
| **1** | Trouser | 98.70 | 13 |
| **2** | Pullover | 91.40 | 86 |
| **3** | Dress | 92.00 | 80 |
| **4** | Coat | 85.50 | 145 |
| **5** | Sandal | 97.90 | 21 |
| **6** | **Shirt** | **79.40** | **206** |
| **7** | Sneaker | 98.80 | 12 |
| **8** | Bag | 98.60 | 14 |
| **9** | Ankle boot | 96.10 | 39 |

* **Best-Performing Classes:** **Sneaker (7)**, **Trouser (1)**, and **Bag (8)** all achieved accuracy close to 99%.
* **Most Difficult Class:** **Shirt (6)** has the lowest accuracy at **79.40%**.

#### Most Confused Pairs (Top 5)

| Rank | True Label (Is) | Predicted as (Was mistaken for) | Count |
| :--- | :--- | :--- | :--- |
| **1.** | **Shirt (6)** | T-shirt/top (0) | 82 times |
| **2.** | **T-shirt/top (0)** | Shirt (6) | 77 times |
| **3.** | **Coat (4)** | Pullover (2) | 70 times |
| **4.** | **Shirt (6)** | Pullover (2) | 55 times |
| **5.** | **Coat (4)** | Shirt (6) | 53 times |

---
###  Future Work

1.  **Data Augmentation:** Implement advanced image transformations (e.g., `RandomRotation`, `ColorJitter`) to improve the model's robustness and generalization, especially for difficult classes like 'Shirt'.
2.  **Model Architecture:** Experiment with deeper architectures or different kernel sizes to potentially capture more complex features.
3.  **Hyperparameter Tuning:** Use optimization techniques to find the optimal combination of learning rate and regularization parameters.
