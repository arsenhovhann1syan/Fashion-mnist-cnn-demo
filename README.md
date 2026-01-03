# Fashion MNIST Classification with PyTorch CNN

## Project Overview

Deep learning image classification project using **Convolutional Neural Networks (CNN)** to identify clothing items from the Fashion MNIST dataset. Built a custom CNN architecture in PyTorch that achieves **93.04% accuracy** on 10 fashion categories.

## Dataset

* **Source:** [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) (PyTorch built-in)
* **Training Set:** 60,000 grayscale images (28×28 pixels)
* **Test Set:** 10,000 grayscale images
* **Classes:** 10 clothing categories
* **Challenge:** Similar visual features across categories (e.g., Shirt vs T-shirt)

### Class Distribution

| Label | Category    | Training Samples | Test Samples |
| ----- | ----------- | ---------------- | ------------ |
| 0     | T-shirt/top | 6,000            | 1,000        |
| 1     | Trouser     | 6,000            | 1,000        |
| 2     | Pullover    | 6,000            | 1,000        |
| 3     | Dress       | 6,000            | 1,000        |
| 4     | Coat        | 6,000            | 1,000        |
| 5     | Sandal      | 6,000            | 1,000        |
| 6     | Shirt       | 6,000            | 1,000        |
| 7     | Sneaker     | 6,000            | 1,000        |
| 8     | Bag         | 6,000            | 1,000        |
| 9     | Ankle boot  | 6,000            | 1,000        |

The dataset is perfectly balanced with no class imbalance issues.

---

## Model Architecture

### Custom CNN Design

```
INPUT LAYER
28×28 grayscale image
    ↓
CONVOLUTIONAL BLOCK 1
Conv2D(1 → 32 filters, kernel=3×3, padding=1) + ReLU
Conv2D(32 → 64 filters, kernel=3×3, padding=1) + ReLU
    ↓
POOLING & REGULARIZATION
MaxPool2D(2×2) → 14×14 feature maps
Dropout(p=0.25)
    ↓
FLATTEN
64 × 14 × 14 = 12,544 features
    ↓
FULLY CONNECTED LAYERS
FC1(12,544 → 128) + ReLU
Dropout(p=0.5)
FC2(128 → 10)
    ↓
OUTPUT
Softmax → 10 class probabilities
```

### Architecture Decisions

| Component    | Choice      | Rationale                      |
| ------------ | ----------- | ------------------------------ |
| Conv Filters | 32 → 64     | Progressive feature extraction |
| Kernel Size  | 3×3         | Standard for small images      |
| Activation   | ReLU        | Prevents vanishing gradients   |
| Pooling      | MaxPool 2×2 | Reduces spatial dimensions     |
| Dropout      | 25% → 50%   | Prevents overfitting           |
| FC Size      | 128 neurons | Balances capacity and speed    |

### Model Summary

```
Total Parameters: 1,663,946
Trainable Parameters: 1,663,946
Model Size: ~6.35 MB
```

---

## Training Configuration

### Hyperparameters

```python
INPUT_SIZE = 28 × 28 = 784
HIDDEN_SIZE = 256 (initial concept, replaced by CNN)
NUM_CLASSES = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
OPTIMIZER = Adam
LOSS_FUNCTION = CrossEntropyLoss
DEVICE = CPU (training took ~30 minutes)
```

### Data Preprocessing

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

Normalizing to [-1,1] helps faster convergence, reduces internal covariate shift, and is standard for grayscale images.

### Training Strategy

* Early Stopping: Not implemented (trained full 20 epochs)
* Learning Rate: Fixed
* Batch Processing: 128 images per batch
* Validation: Evaluated every epoch
* Loss Tracking: Logged every 100 iterations

---

## Results

### Overall Performance

| Metric              | Value       | Benchmark |
| ------------------- | ----------- | --------- |
| Best Test Accuracy  | 93.04%      | Epoch 17  |
| Final Test Accuracy | 92.69%      | Epoch 20  |
| Lowest Test Loss    | 0.2126      | Epoch 9   |
| Final Train Loss    | 0.0952      | Epoch 20  |
| Final Test Loss     | 0.2549      | Epoch 20  |
| Training Time       | ~30 minutes | CPU       |

### Training Progression

| Epoch | Train Loss | Test Loss | Test Accuracy | Status    |
| ----- | ---------- | --------- | ------------- | --------- |
| 1     | 0.5127     | 0.3241    | 88.23%        | Baseline  |
| 5     | 0.2018     | 0.2326    | 91.40%        | Improving |
| 10    | 0.1245     | 0.2203    | 92.30%        | Plateau   |
| 17    | 0.1089     | 0.2353    | 93.04%        | Best      |
| 20    | 0.0952     | 0.2549    | 92.69%        | Final     |

The model started overfitting after epoch 17 (train loss decreased, test loss increased).

---

## Per-Class Performance Analysis

### Accuracy Breakdown

| Class       | Accuracy | Correct/Total | Errors | Status            |
| ----------- | -------- | ------------- | ------ | ----------------- |
| Trouser     | 98.7%    | 987/1000      | 13     | Excellent         |
| Sneaker     | 98.8%    | 988/1000      | 12     | Excellent         |
| Bag         | 98.6%    | 986/1000      | 14     | Excellent         |
| Sandal      | 97.9%    | 979/1000      | 21     | Excellent         |
| Ankle boot  | 96.1%    | 961/1000      | 39     | Good              |
| Dress       | 92.0%    | 920/1000      | 80     | Good              |
| Pullover    | 91.4%    | 914/1000      | 86     | Good              |
| T-shirt/top | 88.5%    | 885/1000      | 115    | Fair              |
| Coat        | 85.5%    | 855/1000      | 145    | Fair              |
| Shirt       | 79.4%    | 794/1000      | 206    | Needs Improvement |

### Most Confused Class Pairs

| True Label  | Predicted As | Count | % of True Class |
| ----------- | ------------ | ----- | --------------- |
| Shirt       | T-shirt/top  | 82    | 8.2%            |
| T-shirt/top | Shirt        | 77    | 7.7%            |
| Coat        | Pullover     | 70    | 7.0%            |
| Shirt       | Pullover     | 55    | 5.5%            |
| Coat        | Shirt        | 53    | 5.3%            |

**Insights:**

* Confusion mostly occurs between upper-body garments (Shirt, T-shirt, Coat, Pullover)
* Lower-body garments and accessories are clearly separated
* Improving texture and fabric feature extraction could reduce upper-body errors

---

## Visualizations

* Training curves: Train loss decreases steadily; test loss U-shaped (overfitting signal after epoch 17)
* Accuracy curves: Rapid improvement epochs 1-10, plateau epochs 10-17
* Confusion matrix: Diagonal dominance; hot spots at Shirt ↔ T-shirt
* Sample predictions: High confidence for clear images; low confidence for ambiguous images
* Misclassified examples: Dark-colored or unusual-perspective garments

---

## Technical Highlights

* Live training visualization using `clear_output` to monitor convergence
* Metrics tracked beyond accuracy: per-class accuracy, confusion matrix, confidence analysis
* Error categorization: systematic, random, and ambiguous errors

---

## Technologies Used

| Category      | Tools               | Version |
| ------------- | ------------------- | ------- |
| Framework     | PyTorch             | 2.0+    |
| Data Loading  | torchvision         | 0.15+   |
| Numerical     | NumPy               | 1.24+   |
| Visualization | Matplotlib, Seaborn | Latest  |
| Metrics       | scikit-learn        | 1.3+    |
| Environment   | Jupyter Notebook    | —       |
| Hardware      | CPU                 | —       |

---

## Project Structure

```
fashion-mnist-cnn/
│
├── README.md
├── fashion_mnist_cnn.ipynb
│
├── outputs/
│   ├── best_model.pth
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   └── misclassified_examples.png
│
└── requirements.txt
```

---

## How to Run

### Installation

```bash
git clone <your-repo-url>
cd fashion-mnist-cnn
pip install torch torchvision matplotlib seaborn scikit-learn jupyter
```

### Execution

```bash
jupyter notebook fashion_mnist_cnn.ipynb
# Run all cells
```

### Using Pre-trained Model

```python
import torch
from model import FashionCNN

model = FashionCNN(num_classes=10)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

with torch.no_grad():
    outputs = model(test_images)
    _, predictions = torch.max(outputs, 1)
```

---

## Key Learnings

* **CNN Architecture Design:** Convolution, pooling, dropout, fully connected layers
* **Training Dynamics:** Overfitting detection, early stopping strategy, learning rate effects
* **Error Analysis:** Confusion matrix interpretation, per-class debugging
* **PyTorch:** Model definition, training loop, device management
* **Domain Insights:** Visual similarity affects model confusion; shape is more important than texture; class balance helps; 28×28 resolution is sufficient for simple classification

---

## Future Improvements

### Model Architecture

* Data augmentation (rotation, flip, translation)
* Deeper network with additional conv layers, batch normalization, skip connections
* Advanced architectures: ResNet18, EfficientNet, ViT-Tiny

### Training Optimizations

* Learning rate scheduling
* Early stopping
* Mixed precision training on GPU

### Advanced Analysis

* Grad-CAM for visualizing important regions
* t-SNE/UMAP embeddings to visualize feature space
* Ensemble models for boosting accuracy

### Deployment

* Quantization for smaller/faster models
* API endpoint using FastAPI
* Mobile deployment via TorchScript or ONNX

---

## Comparison with Benchmarks

| Approach                  | Accuracy | Parameters |
| ------------------------- | -------- | ---------- |
| Logistic Regression       | 84.2%    | ~7K        |
| 2-layer MLP               | 87.5%    | ~200K      |
| LeNet-5                   | 89.8%    | ~60K       |
| Custom CNN (this project) | 93.04%   | 1.66M      |
| ResNet18                  | 94.5%    | ~11M       |
| EfficientNet-B0           | 95.2%    | ~5M        |
| Ensemble (5 models)       | 96.1%    | ~8M        |

---

## Experimental Results

### Ablation Study

| Variant          | Change                     | Accuracy | Δ     |
| ---------------- | -------------------------- | -------- | ----- |
| Full Model       | —                          | 93.04%   | —     |
| No Dropout       | Remove both dropout layers | 91.2%    | -1.8% |
| Single Conv      | Remove Conv2 layer         | 89.7%    | -3.3% |
| Smaller FC       | FC(128) → FC(64)           | 92.1%    | -0.9% |
| No Normalization | Remove normalization       | 88.5%    | -4.5% |

Normalization, dropout, and the second convolution layer significantly contribute to performance.

---

## References & Resources

* [Fashion-MNIST Paper](https://arxiv.org/abs/1708.07747)
* [LeCun CNN Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [Stanford CS231n](http://cs231n.stanford.edu/)
* [Fast.ai Practical Deep Learning](https://course.fast.ai/)

---

## Contact

**Author:** [Arsen Hovhannisyan]
**Email:** [hovhannisyanarsen225@gmail.com)

