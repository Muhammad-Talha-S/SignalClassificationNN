from pathlib import Path
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

def create_training_data(source_1_fp, source_2_fp):
    source_1_df = pd.read_excel(source_1_fp, header=None)
    source_2_df = pd.read_excel(source_2_fp, header=None)

    column_names = [f'DP-{i}' for i in range(1, 3401)]

    source_1_df = source_1_df.drop(source_1_df.columns[:6], axis=1)
    source_1_df.columns = column_names
    source_1_df['source'] = 0 # 0 for source 1

    source_2_df = source_2_df.drop(source_2_df.columns[:6], axis=1)
    source_2_df.columns = column_names
    source_2_df['source'] = 1 # 1 for source 2

    dataframe = pd.concat([source_1_df, source_2_df], axis=0)
    dataframe = dataframe.reset_index(drop=True)
    # dataframe.to_excel('excel_files\dataframe.xlsx', index=False)

    output_param = dataframe['source'].to_numpy()
    input_param = dataframe.drop(columns=['source']).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(input_param, output_param, test_size=0.3, random_state=42,
                                                        stratify=output_param)
    x_train = x_train.reshape(x_train.shape[0], 3400, 1)
    x_test = x_test.reshape(x_test.shape[0], 3400, 1)

    return x_train, y_train, x_test, y_test

def train_and_save_nn_model(x_train, y_train, model_fp):
    # Build the DNN model
    dnn_model = Sequential()

    # Flatten the input signal
    dnn_model.add(Flatten(input_shape=(3400, 1)))

    # First fully connected layer
    dnn_model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Dropout(0.4))

    # Second fully connected layer
    dnn_model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Dropout(0.4))

    # Third fully connected layer
    dnn_model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Dropout(0.3))

    # Fourth fully connected layer
    dnn_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Dropout(0.3))

    # Output layer for binary classification
    dnn_model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    dnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss='binary_crossentropy', metrics=['accuracy'])

    # Implement early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = dnn_model.fit(x_train, y_train, validation_split=0.2, epochs=300, batch_size=32, callbacks=[early_stopping])
    dnn_model.save(model_fp)
    return dnn_model

def train_and_save_cnn_model():
    pass

def train_and_save_lstm_model():
    pass

def main():
    source_1_fp = Path("excel_files\source_1.xlsx")
    source_2_fp = Path("excel_files\source_2.xlsx")
    model_fp = 'nn_models\dnn_model.h5'

    x_train, y_train, x_test, y_test = create_training_data(source_1_fp, source_2_fp)
    dnn_model = train_and_save_nn_model(x_train, y_train, model_fp)
    train_and_save_cnn_model()


if __name__ == "__main__":
    main()