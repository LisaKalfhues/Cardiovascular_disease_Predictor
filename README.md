# Cardiovascular disease Predictor
A binary classification machine learning model to predict heart conditions based on a heart dataset combining common clinical features

## About the project
During the project the dataset was firstly used to perfom data analytics and secondly used to implement and evaluate a machine learning model to predict the occurence of a heart failure based on clinical features from patients.

## The Data
###### *Used data sets: Heart Failure Prediction Dataset from kaggle.com*

This dataset was created by combining different datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features

https://www.kaggle.com/fedesoriano/heart-failure-prediction

## The Model
This ML model was build, trained and evaluated on the following steps:
- Preprocessing: handling categorical variables using *ColumnTransformer*
- Data Scaling: bring features on a relatively similar scale and close to normal distribution using *MinMaxScaler* and *StandardScaler*
- Feature selection:
- Hyperparameter selection: using *GridSearchCV* to find best parameters for different base models
- Stacking: using *StackingClassifier* to combine the output of multiple base estimators/meta classifier to make predictions
- Model training and scoring of predictions

###### *Used libraries: Pandas, Numpy, Scikit-learn, Seaborn, Matplotlib, ydata_profiling*



## Credits



----------------------------------------------------------------
