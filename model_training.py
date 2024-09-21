from pathlib import Path
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

def create_training_data(source_1_fp, source_2_fp):
    print(f"Loading data from {source_1_fp} and {source_2_fp}...")
    source_1_df = pd.read_excel(source_1_fp, header=None)
    source_2_df = pd.read_excel(source_2_fp, header=None)

    column_names = [f'DP-{i}' for i in range(1, 3401)]

    print("Preprocessing source 1 data...")
    source_1_df = source_1_df.drop(source_1_df.columns[:6], axis=1)
    source_1_df.columns = column_names
    source_1_df['source'] = 0  # 0 for source 1

    print("Preprocessing source 2 data...")
    source_2_df = source_2_df.drop(source_2_df.columns[:6], axis=1)
    source_2_df.columns = column_names
    source_2_df['source'] = 1  # 1 for source 2

    dataframe = pd.concat([source_1_df, source_2_df], axis=0)
    dataframe = dataframe.reset_index(drop=True)
    print("Data concatenated and shuffled.")

    output_param = dataframe['source'].to_numpy()
    input_param = dataframe.drop(columns=['source']).to_numpy()

    print("Splitting data into training and testing sets...")
    x_train, x_test, y_train, y_test = train_test_split(input_param, output_param, test_size=0.3, random_state=42,
                                                        stratify=output_param)
    x_train = x_train.reshape(x_train.shape[0], 3400, 1)
    x_test = x_test.reshape(x_test.shape[0], 3400, 1)

    print(f"Data split complete: {x_train.shape[0]} training samples and {x_test.shape[0]} test samples.")
    return x_train, y_train, x_test, y_test

def plot_model_metrices(model_name, model_history):
    output_dir = Path('model_metrices_plots')
    if not output_dir.exists():
        output_dir.mkdir()

    print(f"Plotting metrics for {model_name}...")
    # Plot training & validation accuracy values
    plt.plot(model_history['accuracy'])
    plt.plot(model_history['val_accuracy'])
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{output_dir}/{model_name}_accuracy.png')

    # Plot training & validation loss values
    plt.plot(model_history['loss'])
    plt.plot(model_history['val_loss'])
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{output_dir}/{model_name}_loss.png')

    print(f"Metrics plotted and saved for {model_name}.")

def train_and_save_cnn_model(x_train, y_train, model_fp):
    print("Building CNN model...")
    cnn_model = Sequential()

    cnn_model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(3400, 1)))
    cnn_model.add(MaxPooling1D(pool_size=2))

    cnn_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))

    cnn_model.add(Conv1D(filters=256, kernel_size=5, activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))

    cnn_model.add(Flatten())

    cnn_model.add(Dense(128, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(64, activation='relu'))

    # Output layer for binary classification
    cnn_model.add(Dense(1, activation='sigmoid'))

    print("Compiling CNN model...")
    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='binary_crossentropy', metrics=['accuracy'])

    print("Training CNN model...")
    cnn_history = cnn_model.fit(x_train, y_train, epochs=20, batch_size=16, validation_split=0.2)
    cnn_model.save(model_fp)
    print(f"CNN model saved to {model_fp}.")
    
    plot_model_metrices("Convolutional Neural Network", cnn_history.history)
    return cnn_model

def train_and_save_nn_model(x_train, y_train, model_fp, model_name="Deep Neural Network"):
    print(f"Building {model_name} model...")
    dnn_model = Sequential()

    dnn_model.add(Flatten(input_shape=(3400, 1)))

    dnn_model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Dropout(0.4))

    dnn_model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Dropout(0.4))

    dnn_model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Dropout(0.3))

    dnn_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Dropout(0.3))

    dnn_model.add(Dense(1, activation='sigmoid'))

    print(f"Compiling {model_name} model...")
    dnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print(f"Training {model_name} model...")
    dnn_history = dnn_model.fit(x_train, y_train, validation_split=0.2, epochs=150, batch_size=32, callbacks=[early_stopping])
    dnn_model.save(model_fp)
    print(f"{model_name} model saved to {model_fp}.")

    plot_model_metrices(model_name, dnn_history.history)
    return dnn_model

def train_and_save_lstm_model(x_train, y_train, model_lstm_fp):
    print("Training and saving LSTM model...")
    lstm_model = train_and_save_nn_model(x_train, y_train, model_lstm_fp, model_name="Long-Short Term Memory")
    return lstm_model

def main():
    source_1_fp = Path("excel_files/source_1.xlsx")
    source_2_fp = Path("excel_files/source_2.xlsx")
    model_cnn_fp = 'classification_models/source_classification_using_cnn.h5'
    model_dnn_fp = 'classification_models/source_classification_using_dnn.h5'
    model_lstm_fp = 'classification_models/source_classification_using_lstm.h5'

    print("Starting training pipeline...")
    x_train, y_train, x_test, y_test = create_training_data(source_1_fp, source_2_fp)
    cnn_model = train_and_save_cnn_model(x_train, y_train, model_cnn_fp)
    dnn_model = train_and_save_nn_model(x_train, y_train, model_dnn_fp)
    lstm_model = train_and_save_lstm_model(x_train, y_train, model_lstm_fp)

    print("Training complete.")

if __name__ == "__main__":
    main()
