# WideBot-ML-Classification-Task

## Requirements

Given the training and validation datasets, http://bit.ly/widebot-new-binclf-data , Create and train a machine learning model using the training set that performs well on the validation set.

## Classifier API
Check the API [through this link](https://widebot-classifier.herokuapp.com/)



## Runing the app on your local machine

### Installing

1. To clone the git repository: `https://github.com/ahmedhussiien/WideBot-ML-Classification-Task.git`
2. Install all the requirements: `pip install -r requirements.txt`

### Runing the program

1. Run the app `python app.py`
2. Go to http://127.0.0.1:5000/ or you can get the port that the flask app is running on from the terminal.

## Project structure

### File structure

```
- templates
| - home.html

- data
|- training.csv
|- validation.csv

- models
|- classifier.py
|- classifier.pkl  # saved model
|- data.pkl
|- scaler.pkl

- notebooks
|- Binary_classification.ipynb

- app.py
- serve.py
- requirements.txt
- README.md
```

Note that: if you removed any of of the pretrained model files -files with extension .pkl- the app will automatically train the model again and saves it on disk.
