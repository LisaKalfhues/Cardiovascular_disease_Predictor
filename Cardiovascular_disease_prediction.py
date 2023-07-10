# Import tools
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import ydata_profiling as pp
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning



# Import data
df = pd.read_csv("heart.csv")
print(df.shape)



# EDA using pandas ProfileReport
from ydata_profiling import ProfileReport
profile = pp.ProfileReport(df)
profile.to_file("report.html")



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

# Drop selected features
df_final = df[df.columns.drop(['remainder__HeartDisease','remainder__RestingBP','encoder__RestingECG_LVH'])]
target = df['remainder__HeartDisease']



# Model training
# Train-test-split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_final, target, test_size = 0.20, random_state = 42)

# Find the best hyperparameter of base models
from sklearn.model_selection import GridSearchCV

# SVM
from sklearn.svm import NuSVC, SVC
svm = SVC()
param_grid_svm = {'C': [0.1, 1, 10, 100, 1000], # Regularization parameter, strength of the regularization is inversely prop. to C
              'gamma': ['auto','scale'], # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
              'kernel': ['linear','poly','rbf','sigmoid'] # Specifies the kernel type to be used in the algorithm
             }

gs_svm = GridSearchCV(  estimator = svm,
                        param_grid = param_grid_svm,
                        scoring = "roc_auc",
                        refit = True,
                        verbose = 1,
                        n_jobs = -1)

# Apply to trainings data and print best hyperparameter for Support Vector Machine
gs_svm.fit(x_train,y_train)
print('Best parameters for Support Vector Machine:', gs_svm.best_params_)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
param_grid_knn = { 'n_neighbors' : [2,5,10,15,20], # Number of neighbors to use by default for kneighbors queries
               'weights' : ['uniform','distance'], # Weight function used in prediction, (uniform= all equally, distance=by the inverse of distance)
               'metric' : ['minkowski','euclidean','manhattan']} # Metric to use for distance computation

gs_knn = GridSearchCV(  estimator = knn,
                        param_grid = param_grid_knn,
                        scoring = "roc_auc",
                        refit = True,
                        verbose = 1,
                        n_jobs = -1)

# Apply to trainings data and print best hyperparameter for K-neirest neighbors
gs_knn.fit(x_train,y_train)
print('Best parameters for K-nearest Neighbors:', gs_knn.best_params_)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
param_grid_rfc = { 
    'n_estimators': [100, 150, 300, 500 ], # The number of trees in the forest
    'max_features': ['auto', 'sqrt', 'log2'], # The number of features to consider when looking for the best split
    'max_depth' : [4,5,6,7,8], # The maximum depth of the tree
    'criterion' :['gini', 'entropy'] # The function to measure the quality of a split
}

gs_rfc = GridSearchCV(  estimator = rfc,
                        param_grid = param_grid_rfc,
                        scoring = "roc_auc",
                        refit = True,
                        verbose = 1,
                        n_jobs = -1)

# Apply to trainings data and print best hyperparameter for Random Forest Classifier
gs_rfc.fit(x_train,y_train)
print('Best parameters for Random Forest Classifier:', gs_rfc.best_params_)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'max_depth': [2, 3, 5, 10, 20], # The maximum depth of the tree
    'min_samples_leaf': [5, 10, 20, 50, 100], # The minimum number of samples required to split an internal node
    'criterion': ["gini", "entropy"] # The function to measure the quality of a split
}

gs_dt = GridSearchCV(  estimator = dt,
                        param_grid = param_grid_dt,
                        scoring = "roc_auc",
                        refit = True,
                        verbose = 1,
                        n_jobs = -1)

# Apply to trainings data and print best hyperparameter for Decision Tree Classifier
gs_dt.fit(x_train,y_train)
print('Best parameters for Decision Tree:', gs_dt.best_params_)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
param_grid_nb = {'var_smoothing': np.logspace(0,-9, num=100)} # Portion of the largest variance of all features that is added to variances for calculation stability.

gs_nb = GridSearchCV(  estimator = nb,
                        param_grid = param_grid_nb,
                        scoring = "roc_auc",
                        refit = True,
                        verbose = 1,
                        n_jobs = -1)

# Apply to trainings data and print best hyperparameter for Naive Bayes
gs_nb.fit(x_train,y_train)
print('Best parameters for Naive Bayes:', gs_nb.best_params_)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
param_grid_lr = {
    'penalty' : ['l1', 'l2', 'elasticnet', 'none'], # Specify the norm of the penalty
    'C' : np.logspace(-4, 4, 20), # Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'], # Algorithm to use in the optimization problem
    'max_iter' : [1,2,5,10,50,100] # Maximum number of iterations taken for the solvers to converge
    }

gs_lr = GridSearchCV(  estimator = lr,
                        param_grid = param_grid_lr,
                        scoring = "roc_auc",
                        refit = True,
                        verbose = 1,
                        n_jobs = -1)

# Apply to trainings data and print best hyperparameter for Logistic Regression
gs_lr.fit(x_train,y_train)
print('Best parameters for Logistic Regression:', gs_lr.best_params_)



# StackingClassifier (using best hyperparameters)
from sklearn.ensemble import StackingClassifier

# Define base estimators
base_estimator =(('svm', SVC(C=1, gamma='auto', kernel='linear')),
                ('knn', KNeighborsClassifier(metric='minkowski', n_neighbors=15, weights='distance')),
                ('rfc', RandomForestClassifier(criterion='entropy', max_depth=4, max_features='sqrt', n_estimators=150)),
                ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=10)),
                ('lr', LogisticRegression(C=0.615848211066026, max_iter=1, penalty='l2', solver='saga'))
                )
    
# Define meta classifier
meta_classifier = GaussianNB(var_smoothing=0.1)

# Define stacking ensemble
model = StackingClassifier(estimators=base_estimator,
                            final_estimator=meta_classifier,
                            cv=10,
                            passthrough=True,
                            stack_method='auto'
                            )

model.fit(x_train, y_train)

print('Score for training data:', model.score(x_train, y_train))
print('Score for test data:', model.score(x_test, y_test))