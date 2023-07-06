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

categorical_features = ['encoder__Sex_F',
 'encoder__Sex_M',
 'encoder__ChestPainType_ASY',
 'encoder__ChestPainType_ATA',
 'encoder__ChestPainType_NAP',
 'encoder__ChestPainType_TA',
 'encoder__RestingECG_LVH',
 'encoder__RestingECG_Normal',
 'encoder__RestingECG_ST',
 'encoder__ExerciseAngina_N',
 'encoder__ExerciseAngina_Y',
 'encoder__FastingBS_0',
 'encoder__FastingBS_1',
 'encoder__ST_Slope_Down',
 'encoder__ST_Slope_Flat',
 'encoder__ST_Slope_Up']

numerical_features = ['remainder__Age',
 'remainder__RestingBP',
 'remainder__MaxHR',
 'remainder__Cholesterol',
 'remainder__Oldpeak']



# Feature selection for categorical features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

features_cat = df.loc[:, categorical_features]
target = df['remainder__HeartDisease']

best_features_cat = SelectKBest(score_func = chi2,k = 'all')
fit = best_features_cat.fit(features_cat,target)

# Visualization
featureScores = pd.DataFrame(data = fit.scores_,index = list(features_cat.columns),columns = ['Chi Squared Score']) 

plt.subplots(figsize = (5,5))
sns.heatmap(featureScores.sort_values(ascending = False,by = 'Chi Squared Score'),annot = True,linewidths = 0.6,linecolor = 'black',fmt = '.2f')
plt.title('Selection of Categorical Features')



# Feature selection for numerical features
from sklearn.feature_selection import f_classif

features_num = df.loc[:, numerical_features]
target = df['remainder__HeartDisease']

best_features_num = SelectKBest(score_func = f_classif,k = 'all')
fit = best_features_num.fit(features_num,target)

#Visualization
featureScores = pd.DataFrame(data = fit.scores_,index = list(features_num.columns),columns = ['ANOVA Score']) 

plt.subplots(figsize = (5,5))
sns.heatmap(featureScores.sort_values(ascending = False,by = 'ANOVA Score'),annot = True,linewidths = 0.6,linecolor = 'black',fmt = '.2f');
plt.title('Selection of Numerical Features')



