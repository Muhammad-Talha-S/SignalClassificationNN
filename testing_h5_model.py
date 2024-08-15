from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model from the file
model = load_model('source_classification_using_cnn.h5', compile=False)

# Testing on Excel File

test_file = Path("test_file.xlsx")
test_df = pd.read_excel(test_file, header=None)
total_rows = test_df.shape[0]
total_columns = test_df.shape[1]
signal_columns = 3400

if total_columns > signal_columns:
    columns_to_drop = total_columns - signal_columns 
    test_df = test_df.drop(test_df.columns[:columns_to_drop], axis=1)
    print(f"Dropping {columns_to_drop} to retrive only signal data.")
elif total_columns == signal_columns:
    print("Signal data provided in excel file, no need to remove any columns.")
else:
    print("Incorrect signal data provided. please provide signal having 3400 sampled points.")


for counter in range(total_rows):
    sample = test_df.iloc[counter].to_numpy()
    # Reshape the sample to match the input shape of the model
    sample_reshaped = sample.reshape(1, 3400, 1)
    # Predict the class (returns a probability)
    prediction = model.predict(sample_reshaped)
    if prediction > 0.5:
        source = 1
    else:
        source = 0
    print(f'Predicted Source {source}')