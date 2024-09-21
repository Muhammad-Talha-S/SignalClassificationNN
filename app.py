import pandas as pd
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def click_submit():
    if request.method == 'POST':
        file = request.files['file'] # grab the file
        selected_model = request.form.get('model')

        dict_models_paths = {
        'Neural Network (NN)': r'classification_models\source_classification_using_dnn.h5',
        'Convolutional neural network (CNN)': r'classification_models\source_classification_using_cnn.h5',
        'Long short-term memory (LSTM)': r'classification_models\source_classification_using_cnn.h5' # TODO: LSTM
        }

        model_metrices = {
        'Neural Network (NN)': {

            'confusion_matrix': {
                'TP': 15,
                'FP': 0,
                'TN': 15,
                'FN': 0
            },
            'metrics': {
                'accuracy': 100,
                'precision': 100,
                'recall': 100,
                'f1_score': 100
            }   
        },
        'Long short-term memory (LSTM)': {

            'confusion_matrix': {
                'TP': 0,
                'FP': 0,
                'TN': 0,
                'FN': 0
            },
            'metrics': {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }   
        },
        'Convolutional neural network (CNN)': {

            'confusion_matrix': {
                'TP': 15,
                'FP': 0,
                'TN': 15,
                'FN': 0
            },
            'metrics': {
                'accuracy': 100,
                'precision': 100,
                'recall': 100,
                'f1_score': 100
            }
        }
    }

        # Load the saved model from the file
        model = load_model(dict_models_paths[selected_model], compile=False)

        test_df = pd.read_excel(file, header=None)
        total_rows = test_df.shape[0]
        test_df = test_df.drop(test_df.columns[:6], axis=1)

        signals = list()
        signal_names = list()
        signal_legend = list()
        for counter in range(total_rows):
            sample = test_df.iloc[counter].to_numpy()
            sample_reshaped = sample.reshape(1, 3400, 1)
            result = model.predict(sample_reshaped)
            predict_value = round(result[0][0], 2)
            print(predict_value)
            if predict_value > 0.5:
                source = 2
            else:
                source = 1
            
            signals.append(list(sample))
            signal_names.append(f'Sample Number: {counter+1} Predicted Source: {source}')
            signal_legend.append(f'Sample Number: {counter+1}')

        confusion_matrix = model_metrices[selected_model]['confusion_matrix']
        metrics = model_metrices[selected_model]['metrics']
        
        return render_template(
            'main_page.html',
            signals=signals,
            signal_names=signal_names,
            signal_legend=signal_legend,
            confusion_matrix=confusion_matrix,
            metrics=metrics
        )
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
