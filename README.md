# WideBot-ML-Classification-Task

## Requirements

Given the training and validation datasets, http://bit.ly/widebot-new-binclf-data , Create and train a machine learning model using the training set that performs well on the validation set.

## Classifier API
Check the API [through this link](https://widebot-classifier.herokuapp.com/)  
See a request example [here](https://widebot-classifier.herokuapp.com/api?variable1=b&variable2=23,58&variable3=0,000179&variable4=u&variable5=g&variable6=c&variable7=v&variable8=0,54&variable9=f&variable10=f&variable11=0&variable12=t&variable13=g&variable14=136&variable15=1&variable17=1360000&variable19=0)

## Runing the app on your local machine

### Installing

1. To clone the git repository: `git clone git@github.com:ahmedhussiien/WideBot-ML-Classification-Task.git`
2. Install all the requirements: `pip install -r requirements.txt`

### Runing the program

1. Run the app `python app.py`
2. Go to http://127.0.0.1:5000/ or you can get the port that the flask app is running on from the terminal.
3. You can also run in debug mode `python app.py --debug`

## Project structure

### File structure

```
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

- static
|- swagger.json  # swagger-ui documentation

- templates
| - home.html

- app.py
- serve.py
- requirements.txt
- README.md
```

Note that: if you removed any of of the pretrained model files -files with extension .pkl- the app will automatically train the model again and saves it on disk for faster predicitons.
