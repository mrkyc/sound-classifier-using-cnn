# sound-classifier-using-cnn

Classifying sounds using Convolutional Neural Networks (CNNs).

## Table of contents

1. [Overview](#overview)
2. [Model from scratch in PyTorch](#Model-from-scratch-in-PyTorch)
3. [Transfer learning in FastAI](#Transfer-learning-in-FastAI)
4. [Final test and summary](#Final-test-and-summary)

## Overview

The primary objective of this project is to classify 10 chosen sound categories. To achieve this, I will employ two distinct training approaches for subsequent comparison.

The first model will undergo training from scratch in PyTorch, utilizing a simple Convolutional Neural Network (CNN) architecture. Sound processing classes outlined in this article (https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5) will be incorporated to enhance the model's performance.

For the second model, I will leverage transfer learning through the FastAI library. This involves utilizing one of the pre-trained ResNet models and fine-tuning it to adapt to the given sound classification task.

The dataset employed is sourced from the following GitHub repository: https://github.com/karolpiczak/ESC-50. This dataset comprises 50 distinct categories, each with 40 samples, lasting 5 seconds and possessing identical properties such as sample rate, obviating the need for standardization. The selected categories encompass "airplane," "thunderstorm," "vacuum_cleaner," "cat," "chainsaw," "dog," "chirping_birds," "keyboard_typing," "fireworks," and "church_bells."

To generate input data for both CNN models, spectrograms will be derived from each sound sample. These spectrograms will serve as the foundation for training and evaluating the performance of the models.

### Training dataset augmentation

The training data will undergo on-the-fly augmentation during retrieval from the dataloader object, leveraging audio file processing classes from the specified article. The augmentation procedure will consist of two sequential transformations:

- **Time-shifting**: Prior to the creation of the spectrogram, an audio sample will undergo time-shifting. It moves part of the audio sample from the end to the beginning, similar to shifting in a circular linked list.
- **Spectrogram augmentation**: Following the spectrogram's creation, the second augmentation step involves applying two vertical and two horizontal masks.

## Model from scratch in PyTorch

### Dataset splitting

The metadata file of the ESC-50 dataset conveniently includes a **fold** column, facilitating the division of data into 5 folds. Given the constraints on computational resources, cross-validation, though beneficial, is impractical within a reasonable timeframe. Instead, I will leverage the folds to partition the data into training, validation, and test datasets.

The training dataset will encompass files from folds 2, 3, 4, and 5, constituting 80% of the original data. On the other hand, the validation and test datasets will be derived exclusively from fold 1. This fold will be further subdivided into two equal parts, each ensuring a balanced representation of classes. Consequently, the training dataset will incorporate 80% of the data, while the validation and test datasets will each comprise 10%.

### Hyperparameters tuning

To discover the optimal parameters for the model, I will employ the Optuna library, leveraging its capability to conduct a random search. The search space is defined as follows:

- Optimizer: A choice among Adam, RMSProp, or SGD.
- Learning Rate: A floating-point number ranging from 1e-5 to 1e-1, incremented logarithmically.
- Batch Size: Selected from the set {8, 16, 32, 64}.

30 trials will be performed.

### Results

Study statistics:

- Number of finished trials: 30
- Number of pruned trials: 18
- Number of complete trials: 12

Best trial:

- Validation data accuracy: 0.925
- Parameters:
  - Optimizer: RMSprop
  - Learning rate: 0.001683849132799046
  - Batch size: 32

## Transfer learning in FastAI

### Dataset splitting

As before, I will take **fold** number 1 for validation and test data, while the remaining folds will be used for the training data.

### Hyperparameters tuning

Again, to conduct a random search, I will use the Optuna library. This time, the search space is defined as follows:

- Architecture: A choice among resnet18, resnet34, resnet50.
- Batch size: Selected from the set {8, 16, 32, 64, 128}.
- Base learning Rate: A floating-point number ranging from 1e-5 to 1e-1, incremented logarithmically.
- Freeze epochs: The number of epochs during fine-tuning **before** the `.unfreeze()` method, where only the last layer is fine-tuned. This will also be an integer from 1 to 10.

30 trials will be performed.

### Results

Best trial:

- Value: 0.95
- Params:
  - Architecture: resnet34
  - Batch size: 8
  - Base learning rate: 0.01069194003403398
  - Freeze epochs: 2

## Final test and summary

### Model built from scratch

#### Validation data

Accuracy: 0.925

Confusion matrix:  
![image](https://github.com/mrkyc/portfolio-analysis/assets/82812493/f419f08b-9456-48ea-9e37-59be7c281b0a)

#### Test data

Accuracy: 0.925

Confusion matrix:  
![image](https://github.com/mrkyc/portfolio-analysis/assets/82812493/e838efcb-e28d-454e-9871-7a851ed95d2c)

### Model built using transfer learning

#### Validation data

Accuracy: 0.95

Confusion matrix:  
![image](https://github.com/mrkyc/portfolio-analysis/assets/82812493/d0a2133c-2315-4be4-b4f7-bc8177196576)

#### Test data

Accuracy: 0.875

Confusion matrix:  
![image](https://github.com/mrkyc/portfolio-analysis/assets/82812493/91f6f61a-1dce-49db-859b-f9743310364d)

### Summary

The results indicate noticeable similarities between both models considered. While the first model has a smaller number of layers, building it or searching for the optimal architecture may require a significant amount of time. Contrastingly, the second model, leveraging a pre-trained model and tailoring it to one's specific dataset through transfer learning proves to be a more convenient approach. Undoubtedly, using transfer learning with a library like FastAI streamlines the process, enabling the development of a high-performing model with just a few lines of code. However, it's essential to note that this model tends to be significantly larger, requiring more than 80 MB of storage, in contrast to the first model, which requires less than 0.5 MB. While this may not pose a challenge in today's technological landscape, it could potentially be viewed as a drawback.

It's worth mentioning that the validation and test datasets are relatively modest, each containing only four samples per category. Hence, we can assume that the test data, in both cases, has similar accuracy compared to the validation set. The limited size of the datasets may contribute to the observed results between the validation and test sets. It is important to acknowledge that with a larger test dataset, the accuracy results might exhibit more significant variations than those observed so far. However, given the current dataset size, we can reasonably hypothesize that both models have a comparable or very similar level of accuracy.
