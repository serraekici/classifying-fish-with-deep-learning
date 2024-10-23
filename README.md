# classifying-fish-with-deep-learning

### Fish Species Classification Using Deep Learning

#### Overview
This report outlines a workflow for classifying fish species using a deep learning model built with TensorFlow. The project combines image preprocessing, neural networks, and evaluation techniques to effectively identify fish species from a dataset of PNG images.

#### Data Preparation
The dataset consists of images stored in directories, excluding subfolders named "GT." The file paths and species labels are organized into a pandas DataFrame with two columns: 'path' and 'label'. The labels are mapped to numerical values for easier processing.

#### Image Processing
Images are resized to 64x64 pixels and converted to RGB format for consistency. The images are normalized to values between 0 and 1, flattened into 1D vectors, and one-hot encoded. The dataset is split into training (80%) and validation (20%) sets, ensuring a balanced evaluation.

#### Data Augmentation
Data augmentation is applied using Keras' `ImageDataGenerator`, which adds random transformations like rotation, shifting, and brightness changes. This increases data diversity and reduces overfitting.

#### Model Architecture
The neural network is built using Keras’ Sequential API, with three dense layers (1,024, 512, and 256 neurons), ReLU activation, batch normalization, and dropout layers for regularization. It uses softmax for multi-class classification, compiled with the Adam optimizer and categorical cross-entropy.

#### Model Training and Early Stopping
Early stopping is used to prevent overfitting by monitoring validation loss and stopping training if no improvement occurs for 10 epochs. Training can run for up to 100 epochs with a batch size of 32 but may stop earlier based on performance.

#### Performance Evaluation
Accuracy and loss curves for training and validation are plotted to assess learning. The model’s performance is measured on the validation set, showing test accuracy and loss.

#### Predictions and Confusion Matrix
Predictions on the validation set are compared with actual labels, and a confusion matrix is generated to visualize correct and incorrect classifications, highlighting model strengths and areas for improvement.

#### Conclusion
The workflow demonstrates effective fish species classification using deep learning. By integrating data augmentation, regularization, and performance monitoring, the model achieves robust results, making it well-suited for similar image classification tasks.

Dataset used: https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset/data

project link : https://www.kaggle.com/code/serranurekc/notebook62928b545a/notebook

