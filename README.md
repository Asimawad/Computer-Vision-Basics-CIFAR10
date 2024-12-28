# CIFAR-10 CNN Classification Project

## Abstract
This project focuses on implementing and evaluating various Convolutional Neural Network (CNN) architectures to classify images from the CIFAR-10 dataset. Through systematic experimentation, we explore the effects of model complexity, regularization techniques, and data augmentation on classification accuracy. The report details the methodologies, results, and insights gained from the experiments.

## Introduction
The CIFAR-10 dataset is a widely used benchmark in computer vision, containing 60,000 color images across 10 classes. The goal is to classify these images using CNNs. This project investigates the performance of different CNN architectures, from simple to complex, incorporating techniques such as Dropout and Batch Normalization to improve generalization.

## Dataset
The CIFAR-10 dataset consists of:
- 50,000 training images
- 10,000 test images
- 10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

The dataset was split into training (80%) and validation (20%) subsets for model evaluation during training.

## Methods

### Data Preparation
- **Normalization**: Each image was normalized to have a mean of (0.4914, 0.4822, 0.4465) and a standard deviation of (0.2023, 0.1994, 0.2010).
- **Data Augmentation**: Random cropping and horizontal flipping were applied to enhance data diversity.

### Model Architectures
The following CNN architectures were implemented:

1. **SimpleCNN**:
   - Two convolutional layers with ReLU activations.
   - Two max-pooling layers for down-sampling.
   - Three fully connected layers for classification.

2. **ImprovedCNN**:
   - Three convolutional layers with increased filter depth.
   - Max-pooling after every other convolutional layer.
   - Reduced fully connected layers for improved feature extraction.

3. **AsimNet**:
   - Four convolutional layers with increasing filter sizes.
   - Fully connected layers with 512 neurons before the output layer.

4. **BN_AsimNet**:
   - Adds Batch Normalization after each convolutional layer.
   - Includes Dropout (0.25) to prevent overfitting.

### Training Process
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam with a learning rate of 0.001
- **Batch Size**: 64
- **Epochs**: 10 for SimpleCNN and ImprovedCNN, 15 for AsimNet and BN_AsimNet
- **Hardware**: GPU (if available), otherwise CPU

### Evaluation Metrics
- **Accuracy**: Percentage of correctly classified samples.
- **Loss**: Cross-entropy loss computed during training and validation.

## Results

### Validation Performance
| Model           | Validation Accuracy | Validation Loss |
|-----------------|---------------------|-----------------|
| SimpleCNN       | ~51%                | 1.87            |
| ImprovedCNN     | ~65%                | 1.20            |
| AsimNet         | ~72%                | 0.98            |
| BN_AsimNet      | ~77%                | 0.82            |

### Test Performance
| Model           | Test Accuracy | Test Loss |
|-----------------|---------------|-----------|
| BN_AsimNet      | ~77%          | 0.80      |

### Observations
1. **SimpleCNN**: Serves as a baseline but has limited capacity to capture complex patterns.
2. **ImprovedCNN**: Benefits from deeper architectures and more filters.
3. **AsimNet**: Higher capacity leads to better generalization, though prone to overfitting without regularization.
4. **BN_AsimNet**: Batch Normalization and Dropout together provide the best results, balancing performance and generalization.

### Plots
Loss and accuracy plots for each model are saved in the `plots/` directory for further analysis.

## Discussion
- **Regularization**: Dropout stabilizes training and prevents overfitting in larger models.
- **Batch Normalization**: Accelerates training and enhances generalization when combined with Dropout.
- **Data Augmentation**: Improves performance by simulating variations in the dataset.

## Conclusion
The best-performing model, BN_AsimNet, achieved a validation accuracy of ~77% and generalized well on the test set. This highlights the importance of combining deeper architectures with effective regularization techniques.

## Future Work
1. Experiment with advanced architectures like ResNet and VGG.
2. Explore learning rate schedulers and adaptive optimizers.
3. Integrate additional regularization techniques such as L2 weight decay.

## References
1. PyTorch Documentation: https://pytorch.org/docs
2. CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
3. Batch Normalization Paper: https://arxiv.org/abs/1502.03167

---

This report summarizes the systematic exploration of CNN architectures on CIFAR-10, providing insights into the impact of model depth, regularization, and data augmentation.

