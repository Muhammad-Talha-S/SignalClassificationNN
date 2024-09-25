# Signal Classification Using Deep Learning Models

## Project Overview

This project focuses on classifying digital time signals (noise) that originate from two distinct sources using deep learning techniques. The goal is to develop a system that can predict whether a signal was emitted from source #1 or source #2 based on the labeled dataset. Three different models are implemented for signal classification: a simple Neural Network (NN), a Convolutional Neural Network (CNN), and a Long Short Term Memory (LSTM) network.

The models are pre-trained and stored for future use. This project not only trains these models but also provides a Flask-based web application where users can upload signal data, select a model, and view prediction results along with relevant performance metrics.

## Objectives

The primary objectives of this research are:

- To develop and compare the performance of NN, CNN, and LSTM models for signal classification.
- To create pre-trained models for immediate use in practical applications.
- To design and implement a web-based application that allows users to utilize these models for real-time signal classification.

## Technologies Used

1. **Python**
2. **Flask**
3. **HTML, CSS, JavaScript**
4. **TensorFlow**

## Models Implemented

1. **Neural Network (NN)**: A basic feedforward neural network that learns to classify the signals.
2. **Convolutional Neural Network (CNN)**: A model that uses convolutional layers to capture spatial hierarchies in the data, which is particularly effective for image-like or structured input such as time-series signals.
3. **Long Short Term Memory (LSTM)**: A type of recurrent neural network (RNN) that captures temporal dependencies in time-series data, useful for sequential signal prediction.

## Features

- Pre-trained models for immediate use:
  - `source_classification_using_dnn.h5` (Neural Network)
  - `source_classification_using_cnn.h5` (CNN)
  - `source_classification_using_lstm.h5` (LSTM)
- A web application with an easy-to-use interface for signal prediction.
- Performance metrics including confusion matrix, recall, precision, and accuracy.
- Visualization of predicted signal classifications.

## Folder Structure

SignalClassificationNN/ # Root directory of the project
├── classification_models/ # Folder containing saved model files
│ ├── source_classification_using_cnn.h5 # CNN model
│ ├── source_classification_using_dnn.h5 # DNN model
│ ├── source_classification_using_lstm.h5 # LSTM model
├── excel_files/ # Folder containing signal data in Excel format
│ ├── source_1.xlsx # Excel file for signals from source 1
│ ├── source_2.xlsx # Excel file for signals from source 2
├── templates/ # Folder for HTML templates for the web app
│ ├── index.html # Home page template
│ ├── main_page.html # Main classification page template
├── app.py # Flask web app for signal classification
├── data_processing_and_model_preparation.ipynb # Jupyter notebook for data processing and model training
├── LICENSE # License file
├── model_training.py # Python script for training models
├── README.md # Project readme file
└── requirements.txt # Python package dependencies

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Muhammad-Talha-S/SignalClassificationNN.git
cd SignalClassificationNN
```

### Step 2: Install the Required Libraries

```bash
pip install -r requirements.txt
```

### Step 3: Train the Models

Run the python file model_training.py to train the models and generate the pre-trained .h5 files for the Deep Neural Network, Convolutional Neural Network, and Long-Short Term Memory:

```bash
python model_training.py
```

#### Output Models

The following trained models will be saved in the `classification_models` directory:

- `source_classification_using_dnn.h5`
- `source_classification_using_cnn.h5`
- `source_classification_using_lstm.h5`

#### Model Metrics

The command will also generate Model Accuracy and Loss plots, which will be saved in the `model_metrices_plots` directory.

### Step 4: Run the Flask Web Application

To start the web application, run the following command:

```bash
python app.py
```

## Web Application

Once the app is running, it will open on your local machine (typically at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)).

### Web Application Features

After launching the app, the homepage will provide the following options:

- **Upload an Excel file** containing signals for prediction.
- **Select the model** you wish to use (DNN, CNN, or LSTM).
- **Submit the form** to classify the signals.

Upon submission, the app will visualize the results, including:

- Predicted labels
- Confusion matrix
- Recall, precision, and accuracy for the selected model.

### Screenshot of Homepage

_Insert Image Here_

### Predict Page Video

_Insert Video Here_

---

### Group Members

- Muhammad Talha Saleem
- Kashif Hussain
- Naila Shaheen

### Project Submitted To

**Professor Andreas Pech**  
Computational Intelligence Project
