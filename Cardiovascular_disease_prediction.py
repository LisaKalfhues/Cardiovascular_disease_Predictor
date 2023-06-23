# Import tools
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import ydata_profiling as pp

# Import data
df = pd.read_csv("heart.csv")
print(df.shape)

# EDA
from ydata_profiling import ProfileReport
profile = pp.ProfileReport(df)
profile.to_file("report.html")

categorical_features = ['Sex','ChestPainType','RestingECG','ExerciseAngina','FastingBS','ST_Slope','HeartDisease']
numerical_features = ['Age','RestingBP','MaxHR','Cholesterol','Oldpeak']

# Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

## create an object of the ColumnTransformer class
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), ['Sex','ChestPainType','RestingECG','ExerciseAngina','FastingBS','ST_Slope'])], remainder='passthrough')

## fit and transform the dataset
df = np.array(columnTransformer.fit_transform(df), dtype = np.str_)



