import os.path
import numpy as np
import pandas as pd
from models.classifier import MODEL_FILENAME, SCALER_FILENAME, DATA_FOR_PREDICTIONS_FILENAME, train_model, prepare_input, load_model_data

args = ['variable1', 'variable2', 'variable3', 'variable4', 'variable5',
        'variable6', 'variable7', 'variable8', 'variable9', 'variable10',
        'variable12', 'variable13', 'variable15', 'variable18', 'variable11', 
        'variable14', 'variable17', 'variable19']


def get_model_api():
    '''Returns lambda function for api
    '''

    IS_MODEL_ON_DISK = os.path.isfile(MODEL_FILENAME)
    IS_SCALER_ON_DISK = os.path.isfile(SCALER_FILENAME)
    IS_DATA_ON_DISK = os.path.isfile(DATA_FOR_PREDICTIONS_FILENAME)

    if (not IS_MODEL_ON_DISK) or ( not IS_SCALER_ON_DISK) or (not IS_DATA_ON_DISK):
        print("\nModel not found on disk\nTraining model...")
        train_model()
    else:
        print("\nModel found on disk")

    model, scaler,  X_categorical_mode, X_numerical_mean, encoded_features = load_model_data()

    def model_api(request):
        '''Returns predictions given a dictionary for features values
        '''

        df = pd.DataFrame(columns=args, index=[0])
        for arg in args:
            df[arg].iloc[0] = request.args.get(arg)

        X = prepare_input(df, X_categorical_mode, X_numerical_mean, encoded_features, scaler)
        y_pred = model.predict(X)
        y_pred_enc = np.where(y_pred == 0, 'no.', 'yes.')
        return y_pred_enc

    return model_api