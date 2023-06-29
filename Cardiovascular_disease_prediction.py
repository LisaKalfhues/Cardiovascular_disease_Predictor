# Import tools
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import ydata_profiling as pp



# Import data
df = pd.read_csv("heart.csv")
print(df.shape)



# EDA using pandas ProfileReport
from ydata_profiling import ProfileReport
profile = pp.ProfileReport(df)
profile.to_file("report.html")

categorical_features = ['Sex','ChestPainType','RestingECG','ExerciseAngina','FastingBS','ST_Slope','HeartDisease']
numerical_features = ['Age','RestingBP','MaxHR','Cholesterol','Oldpeak']



# Preprocessing usinf ColumnTranformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

## create an object of the ColumnTransformer class
ohe = OneHotEncoder(sparse=False,handle_unknown="ignore")
columnTransformer = ColumnTransformer([('encoder', ohe, ['Sex','ChestPainType','RestingECG','ExerciseAngina','FastingBS','ST_Slope'])], remainder='passthrough')

## fit and transform the dataset
df = columnTransformer.fit_transform(df)
df = pd.DataFrame(df, columns = columnTransformer.get_feature_names_out())



# Data scaling using MinMaxScaler and StandardScaler
from sklearn.preprocessing import MinMaxScaler,StandardScaler

mms = MinMaxScaler()
ss = StandardScaler()

df['remainder__Oldpeak'] = mms.fit_transform(df[['remainder__Oldpeak']]) 
df['remainder__Age'] = ss.fit_transform(df[['remainder__Age']])
df['remainder__RestingBP'] = ss.fit_transform(df[['remainder__RestingBP']])
df['remainder__Cholesterol'] = ss.fit_transform(df[['remainder__Cholesterol']])
df['remainder__MaxHR'] = ss.fit_transform(df[['remainder__MaxHR']])

selected_columns = ['remainder__Oldpeak', 'remainder__Age', 'remainder__RestingBP', 'remainder__Cholesterol', 'remainder__MaxHR']
selected_df = df[selected_columns]
corr_matrix = selected_df.corr()
sns.heatmap(corr_matrix, annot = True)
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')



# Feature selection for categorical features

# Feature selection for numerical features

