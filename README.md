Chromosome Structure Detection using CNN

This project focuses on detecting chromosome structures as either normal or abnormal using a Convolutional Neural Network (CNN). Additionally, it provides a user-friendly web interface where users can drag and drop chromosome images for classification.

Project Workflow

1. Data Collection

The project began by collecting a raw dataset of chromosome images. This dataset consists of labeled images categorized as either normal or abnormal chromosomes.

2. Data Augmentation

To improve the model's generalization and performance, data augmentation techniques were applied, such as:

Rotation

Flipping

Zooming

Cropping

3. Exploratory Data Analysis (EDA)

Performed EDA to understand the dataset distribution and visualize key features. Insights included:

Class distribution

Sample visualizations of augmented images

Pixel intensity histograms

4. Data Splitting

The dataset was split into:

Training set: 70% of the data

Validation set: 15% of the data

Testing set: 15% of the data

5. Model Building

A Convolutional Neural Network (CNN) was designed with the following architecture:

Input Layer: Accepts augmented chromosome images

Convolutional Layers: Feature extraction using convolution filters

Pooling Layers: Max-pooling applied to reduce dimensionality

Fully Connected Layers: For classification into normal or abnormal categories

Output Layer: Softmax activation for binary classification

6. Prediction

The trained model was used to predict the classification of new chromosome images as either normal or abnormal.

7. Model Evaluation

Metrics such as accuracy, precision, recall, and F1-score were calculated to evaluate the performance of the model. Confusion matrices and ROC curves were also plotted for deeper insights.

8. Web Interface

A user-friendly web interface was developed using [Flask].

Features:

Drag-and-drop functionality for uploading images

Displays the uploaded image

Predicts and displays whether the chromosome is normal or abnormal

Installation

Prerequisites

Ensure the following are installed:

Python 3.7+

Required Python libraries 

Results

Accuracy: 89%

Precision: 85%

Recall: 90%

F1-Score: 87%

Future Enhancements

Extend the model to detect specific syndromes associated with abnormal chromosomes.

Improve the web interface for batch image uploads.

Deploy the application to a cloud platform for broader accessibility.

Acknowledgments

Dataset Source**: Provided by our project guide, P. Devika

Frameworks Used:  
  - TensorFlow/Keras for model building.  
  - Flask for the web interface.  

