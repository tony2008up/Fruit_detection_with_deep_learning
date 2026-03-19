# Date Fruit Detection

A deep learning classification project that predicts **date fruit varieties** from image-extracted features using an **Artificial Neural Network (ANN)** built with **TensorFlow / Keras**.

This notebook walks through the full machine learning pipeline: data loading, cleaning, class balance analysis, preprocessing, ANN modeling, hyperparameter experiments, and final model evaluation.

---

## Project Overview

The goal of this project is to classify **7 types of date fruits** based on **34 numerical features** extracted from images.

The workflow includes:

- loading and exploring the dataset
- checking missing values and duplicates
- analyzing class imbalance
- encoding class labels
- scaling features with MinMaxScaler
- splitting data into train, validation, and test sets
- building a multi-layer ANN
- testing multiple hyperparameter settings
- selecting and evaluating the best model

---

## Dataset

The dataset used in this project is **Date_Fruit_Datasets.csv**.

### Target Classes

The notebook classifies the following 7 date fruit varieties:

- BERHI
- DEGLET
- DOKOL
- IRAQI
- ROTANA
- SAFAVI
- SOGAY

### Features

The dataset contains:

- **898 samples**
- **34 numerical input features**
- **1 target column (`Class`)**

These features are image-based measurements such as:

- area
- perimeter
- major axis
- minor axis
- eccentricity
- color statistics
- texture-related properties

---

## Data Cleaning and Exploration

The notebook includes several exploratory and preprocessing steps:

### Data inspection
- previewing the first few rows
- generating descriptive statistics
- checking data types and column structure

### Data cleaning
- checking for missing values
- checking for duplicate rows
- removing duplicates and null values

### Key findings
- no missing values were found
- no duplicate rows were found
- all feature columns are numeric and suitable for ANN training

---

## Class Balance

A class distribution plot shows that the dataset is **imbalanced**.

For example:
- **DOKOL** and **SAFAVI** have relatively high sample counts
- **BERHI** has noticeably fewer samples

To address this, the model uses **class weights** during training so minority classes are not ignored.

---

## Preprocessing

Before training, the notebook performs:

- separation of features and labels
- label encoding for the 7 fruit classes
- MinMax scaling of the feature values
- stratified train / validation / test split

### Label Encoding Mapping

- BERHI → 0
- DEGLET → 1
- DOKOL → 2
- IRAQI → 3
- ROTANA → 4
- SAFAVI → 5
- SOGAY → 6

### Data Split

The processed data is divided into:

- **Training set**
- **Validation set**
- **Test set**

This allows fair model tuning and final performance evaluation.

---

## Model Architecture

The project uses a **Sequential ANN** in TensorFlow/Keras.

### Final selected model
- input layer for 34 features
- hidden layer with **64 neurons**, activation = `tanh`
- hidden layer with **32 neurons**, activation = `tanh`
- output layer with **7 neurons**, activation = `softmax`

### Training setup
- optimizer: **Adam**
- loss function: **sparse_categorical_crossentropy**
- metric: **accuracy**
- early stopping on validation accuracy
- class weights applied to reduce imbalance impact

---

## Hyperparameter Experiments

The notebook compares multiple model configurations, including changes to:

- number of hidden layers
- number of neurons
- activation functions
- optimizer choices
- dropout usage

After testing several variations, the best-performing model was:

**M3_tanh_64_32_adam**

This model achieved the strongest validation performance while maintaining good generalization.

---

## Results

### Best model performance

- **Training Accuracy:** 93.15%
- **Validation Accuracy:** 96.30%
- **Test Accuracy:** 91.53%

### Interpretation

The model performs very well across all data splits.

The slight decrease from validation accuracy to test accuracy is normal and suggests that the model generalizes reasonably well to unseen data.

The confusion matrix indicates that most classes are classified correctly, while some visually similar date varieties are harder to distinguish.

---

## Visualizations Included

The notebook includes the following visual outputs:

- dataset preview and summary statistics
- bar chart of class distribution
- training and validation accuracy / loss trends
- confusion matrices for validation and test predictions
- comparison of multiple hyperparameter experiments

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras

---

## Project Structure

```bash
.
├── Date_fruit_detection.ipynb
├── Date_Fruit_Datasets.csv
└── README.md
```

---

## How to Run

1. Clone this repository.
2. Make sure the dataset file `Date_Fruit_Datasets.csv` is in the same folder as the notebook.
3. Install the required Python libraries.
4. Open the notebook in Jupyter Notebook or Google Colab.
5. Run the cells from top to bottom.

### Example installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

---

## Learning Outcomes

This project demonstrates how to:

- prepare tabular image-derived data for deep learning
- handle imbalanced multi-class classification
- build and evaluate ANN models with TensorFlow
- compare hyperparameter settings systematically
- interpret performance using confusion matrices and accuracy scores

---

## Possible Improvements

Future improvements could include:

- trying deeper neural networks
- using batch normalization
- applying learning rate scheduling
- testing other scalers such as StandardScaler
- using cross-validation for more robust evaluation
- comparing ANN performance with classical models such as Random Forest, SVM, or XGBoost

---

## Author

This project was created as part of a machine learning / deep learning coursework exercise on multi-class classification.

