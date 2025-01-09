Title
Multi-Layer Perceptron Classifier: Iris Dataset

Overview
This project implements a Multi-Layer Perceptron (MLP) classifier to classify the Iris dataset using its first two features for simplicity. The model's decision boundaries and performance are visualized to understand the classifier's behavior.

Features
Trains an MLP classifier with two hidden layers of 10 neurons each.
Uses the Iris dataset, a popular dataset in machine learning.
Includes standardization of input features for effective training.
Visualizes decision boundaries and a confusion matrix to evaluate model performance.
Displays accuracy and a classification report.
Dataset
Source: The Iris dataset is loaded from the sklearn.datasets module.
Features Used: Only the first two features of the dataset:
Sepal Length
Sepal Width
Target Classes: Three species of Iris flowers:
Setosa
Versicolor
Virginica
Code Workflow
Data Loading: Load the Iris dataset and select the first two features for visualization purposes.
Preprocessing: Standardize the training and test datasets using StandardScaler.
Model Training: Train the MLP classifier with:
Two hidden layers with 10 neurons each.
Maximum iterations set to 500.
Model Evaluation:
Calculate and display accuracy and a classification report.
Plot decision boundaries for visualizing the classifier's regions.
Display a confusion matrix heatmap.
Visualization:
Decision boundaries highlight the classifier's performance visually.
Confusion matrix shows the model's classification performance numerically.
