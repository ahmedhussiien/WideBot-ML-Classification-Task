# load and clean the training set 
# train model on the cleaned dataset

import numpy as np
import pandas as pd
import joblib
import os.path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

TRAIN_DATA_PATH = "data/training.csv"
TEST_DATA_PATH = "data/validation.csv"
MODEL_FILENAME = "models/classifier.pkl"
SCALER_FILENAME = "models/scaler.pkl"
DATA_FOR_PREDICTIONS_FILENAME = 'models/data.pkl'


CLASSIFIER_TUNED_ARGS = {'n_estimators': 100,
 'min_samples_split': 10,
 'min_samples_leaf': 1,
 'max_features': 'sqrt',
 'max_depth': 50,
 'bootstrap': True}


def load_data(filename):
    '''Load the data from the input files

    Args:
        filename (str):  categories filename
    Returns:
        df (pandas.DataFrame): dataframe containing the data
    '''

    df = pd.read_csv(filename, sep=';')

    return df


def prepare_training_data(df):
    '''Clean the data
    
    Args:
        df (pandas.DataFrame): dataframe containing the uncleaned data
    
    Returns:
        X : Contains input features
        y : Contains target features
        X_categorical_mode : Contains the mode value for each categorical feature
        X_numerical_mean : Contains the mean value for each numerical feature
        encoded_features : Columns names after encoding categorical features
        scaler : StandardScaler object  

    '''

    # Dropping duplicates and unnecessary features
    df.drop_duplicates(inplace=True)
    df.drop(['variable18', 'variable19', 'variable14', 'variable4', 'variable6'], axis =1, inplace=True)

    # Formatting data
    df['variable3'] = df['variable3'].apply(lambda x: float(x.replace(',', '.')))
    df['variable2'] = df['variable2'].apply(lambda x: float(str(x).replace(',', '.')))
    df['variable8'] = df['variable8'].apply(lambda x: float(x.replace(',', '.')))

    df['classLabel'] = df['classLabel'].replace('no.', 0)
    df['classLabel'] = df['classLabel'].replace('yes.', 1)

    # Handling missing values
    target_feature = 'classLabel'
    categorical_features = df.select_dtypes(exclude=['float64', 'int64']).columns
    numerical_features = df.select_dtypes(include=['float64', 'int64']).drop(target_feature, axis=1).columns

    X = df.drop(target_feature, axis=1)
    X_categorical = X[categorical_features]
    X_numerical = X[numerical_features]

    ## Filling missing numerical values with the mean
    X_numerical_mean = X_numerical.mean()
    X_numerical = X_numerical.fillna(X_numerical_mean)
    df[numerical_features] = X_numerical

    ## Filling missing categorical values with the mode
    X_categorical_mode = X_categorical.mode().iloc[0]
    X_categorical = X_categorical.fillna(X_categorical_mode)
    df[categorical_features] = X_categorical

    # Encoding categorical features
    df_enc = pd.get_dummies(data=df, columns=categorical_features)

    # Splitting data
    y = df_enc[target_feature]
    X = df_enc.drop(target_feature, axis=1)
    encoded_features = X.columns


    # Scaling numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, X_categorical_mode, X_numerical_mean, encoded_features, scaler


def build_model(X, y, do_randomized_search = True):
    '''build a classfier model
    Args:
        do_randomized_search (bool): specify if the function should do randomized search or not
    Returns:
        model: built model
    '''

    if (do_randomized_search):

        n_estimators = [100, 200, 400, 800, 1600, 2000]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap} 

        rf_random = RandomizedSearchCV(estimator =  RandomForestClassifier(), param_distributions = random_grid, n_iter = 100, cv = 3, verbose=False, random_state=42, n_jobs = -1) 
        
        print('\nTraining model 🐎...')
        rf_random.fit(X, y)
        return rf_random.best_estimator_

    else:

        rf_clf = RandomForestClassifier(**CLASSIFIER_TUNED_ARGS)

        print('\nTraining model 🐎...')
        rf_clf.fit(X, y)
        return rf_clf


def save_model_data(model, scaler,  X_categorical_mode, X_numerical_mean, encoded_features):
    '''saves all the needed objects to make new predictions without training again
    '''

    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(scaler, SCALER_FILENAME)

    data = {'X_categorical_mode': X_categorical_mode, 
    'X_numerical_mean' : X_numerical_mean,
    'encoded_features': encoded_features
    }

    joblib.dump(data, DATA_FOR_PREDICTIONS_FILENAME)

    
def load_model_data():
    '''loads all the needed objects to make new predictions without training again
   
    Returns:
        model : Classifier model
        scaler : StandardScaler object 
        X_categorical_mode : Contains the mode value for each categorical feature
        X_numerical_mean : Contains the mean value for each numerical feature
        encoded_features : Columns names after encoding categorical features 
    '''

    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    data = joblib.load(DATA_FOR_PREDICTIONS_FILENAME)

    return model, scaler,  data['X_categorical_mode'], data['X_numerical_mean'], data['encoded_features']


def prepare_input(df, X_categorical_mode, X_numerical_mean, encoded_features, scaler):
    '''prepare new data for prediciton
    '''
    
    # Dropping unnecessary features
    df.drop(['variable18', 'variable19', 'variable14', 'variable4', 'variable6'], axis =1, inplace=True)
    
    # Formatting data
    df['variable3'] = df['variable3'].apply(lambda x: float(x.replace(',', '.')))
    df['variable2'] = df['variable2'].apply(lambda x: float(str(x).replace(',', '.')))
    df['variable8'] = df['variable8'].apply(lambda x: float(x.replace(',', '.')))

    # Categorizing features
    categorical_features = df.select_dtypes(exclude=['float64', 'int64']).columns
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

    # Handling missing values
    df[numerical_features] = df[numerical_features].fillna(X_numerical_mean)
    df[categorical_features] = df[categorical_features].fillna(X_categorical_mode)
    
    # Encoding categorical features
    df_enc = pd.get_dummies(data=df, columns=categorical_features)
    df_enc = df_enc.reindex(columns = encoded_features, fill_value=0)
    
    # Scaling data
    X = pd.DataFrame(scaler.transform(df_enc))
    
    return X


def train_model():
    '''train a classifier and saves it on disk
    '''

    print('\nLoading data ⌛...')
    df = load_data(TRAIN_DATA_PATH)

    print('\nPreparing data 🧹...')
    X, y, X_categorical_mode, X_numerical_mean, encoded_features, scaler = prepare_training_data(df)

    print('\nBuilding model 👷...')
    clf = build_model(X, y)

    print('\nSaving model 💾...')
    save_model_data(clf, scaler, X_categorical_mode, X_numerical_mean, encoded_features)

    print('\nTrained model saved ✅')


def make_predictions(df):

    model, scaler,  X_categorical_mode, X_numerical_mean, encoded_features = load_model_data()
    X = prepare_input(df, X_categorical_mode, X_numerical_mean, encoded_features, scaler)

    y_pred = model.predict(X)
    y_pred_enc = np.where(y_pred == 0, 'no.', 'yes.')

    return y_pred_enc
    

if __name__ == '__main__':

    IS_MODEL_ON_DISK = os.path.isfile(MODEL_FILENAME)
    IS_SCALER_ON_DISK = os.path.isfile(SCALER_FILENAME)
    IS_DATA_ON_DISK = os.path.isfile(DATA_FOR_PREDICTIONS_FILENAME)

    if (not IS_MODEL_ON_DISK) or ( not IS_SCALER_ON_DISK) or (not IS_DATA_ON_DISK):
        print("\nModel not found on disk\nTraining model...")
        train_model()
    else:
        print("\nModel found on disk")

    df = load_data(TEST_DATA_PATH)
    y_true = df['classLabel']
    X = df.drop('classLabel', axis=1)
    y_pred = make_predictions(X)

    print("\nTesting set classification report:\n", classification_report(y_pred, y_true))


    
    
