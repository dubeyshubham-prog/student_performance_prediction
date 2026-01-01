#IMPORT BASIC LIBRARIES FOR THE PROJECT=>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#IMPORT ALL LIBRARIES REQUIRED FOR MODEL TRAINING=>
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

#IMPORTANT CODE SNIPPET TO PRINT THE ENTIRE DATASET:
#SHOWS ALL ROWS
pd.set_option('display.max_rows',None)
#SHOWS ALL COLUMNS
pd.set_option('display.max_columns',None)
#PREVENT LINE WRAPPING
pd.set_option('display.width',None)
#SHOW FULL CONTENT IN EACH CELL
pd.set_option('display.max_colwidth',None)

#ALL THE REQUIRED DATA PREPROCESSING IS DONE HERE=>
df = pd.read_csv('DATA/stud.csv')
'''
print(df.head())
print(df.shape)
'''

#PREPARING FOR TRAINING AND TESTING DATASET=>
#INDEPENDENT VARIABLE:
X = df.drop(columns=['math_score'],axis=1)
#DEPENDENT VARIABLE:
y = df['math_score']

#ANALYSIS OF THE DATASET FOR TRANSFORMATION=>
'''
print("Categories in 'Gender' variable:",end=" " )
print(df['gender'].unique())

print("Categories in 'Race_Ethnicity' variable:",end=" ")
print(df['race_ethnicity'].unique())

print("Categories in 'parental level of education' variable:",end=" " )
print(df['parental_level_of_education'].unique())

print("Categories in 'lunch' variable:",end=" " )
print(df['lunch'].unique())

print("Categories in 'test preparation course' variable:",end=" " )
print(df['test_preparation_course'].unique())
'''

#CREATE COLUMN TRANSFORMER WITH 2 TYPES OF TRANSFORMER=>
num_features = X.select_dtypes(exclude='object').columns
cat_features = X.select_dtypes(include='object').columns
#REQUIRED LIBRARIES:
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
#CREATING OBJECTS:
numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder()
#CREATING PIPELINE:
preprocessor = ColumnTransformer(
    [
        ('OneHotEncoder',oh_transformer,cat_features),
        ('StandardScaler',numeric_transformer,num_features)
    ]
)
#TRANSFORMING THE DATA:
X = preprocessor.fit_transform(X)

#SEPERATE DATASET INTO TRAIN AND TEST
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
# print(X_train.shape)
# print(y_train.shape)

#CREATE AN EVALUATE TRAINING TO GIVE ALL METRICS AFTER MODEL TRAINING=>
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

#CREATING A DICTIONARY OF ALL ALGORITHMS TO TRAIN THE MODEL=>
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
}

#CREATE TWO LIST OBJECTS=>
model_list = []
r2_list = []

#CREATING A LOOP TO ACCESS THE ALL ALGORITHMS=>
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train,y_train)#TRAINING THE MODEL

    #MAKE PREDICTIONS:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    #EVALUATING TRAIN AND TEST DATA:
    #TRAINING DATA:
    model_train_mae, model_train_rmse,model_train_r2 = evaluate_model(y_train,y_train_pred)
    #TEST DATA:
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test,y_test_pred)

    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))
    print('----------------------------------')
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    r2_list.append(model_test_r2)
    print('=' * 35)
    print('\n')

#PRINT R2 SCORE IN THE FORM OF DATAFRAME FOR EVERY MODEL=>
print(pd.DataFrame(zip(model_list,r2_list),
                   columns=['Model Name', 'R2 Score']).sort_values(by=['R2 Score'],ascending=False))
