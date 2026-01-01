# ========================= STUDENT PERFORMANCE INDICATOR (PROJECT MAP) =============================>
# LIFE CYCLE OF MACHINE LEARNING PROJECT ----------------------------->
'''
1 --> UNDERSTANDING THE PROBLEM STATEMENT
2 --> DATA COLLECTION
3 --> DATA CHECKS TO PERFORM
4 --> EXPLORATORY DATA ANALYSIS
5 --> DATA PREPROCESSING
6 --> MODEL TRAINING
7 --> CHOOSE THE BEST ONE
'''
import warnings

from pyexpat import features

# PROBLEM STATEMENT -------------------------------------------------->
'''
THIS PROJECT UNDERSTAND HOW THE STUDENTS PERFORMANCE (TEST SCORE) IS AFFECTED BY OTHER
VARIABLES SUCH AS GENDER, ETHNICITY, PARENTAL LEVEL OF EDUCATION, LUNCH AND TEST PREP-
RATION COURSE. 
'''
# ===================================================================================================>


# ========================= STUDENT PERFORMANCE INDICATOR (PROJECT MAP) =============================>
# LIFE CYCLE OF MACHINE LEARNING PROJECT ----------------------------->
'''
1 --> UNDERSTANDING THE PROBLEM STATEMENT
2 --> DATA COLLECTION
3 --> DATA CHECKS TO PERFORM
4 --> EXPLORATORY DATA ANALYSIS
5 --> DATA PREPROCESSING
6 --> MODEL TRAINING
7 --> CHOOSE THE BEST ONE
'''

# PROBLEM STATEMENT -------------------------------------------------->
'''
THIS PROJECT UNDERSTAND HOW THE STUDENTS PERFORMANCE (TEST SCORE) IS AFFECTED BY OTHER
VARIABLES SUCH AS GENDER, ETHNICITY, PARENTAL LEVEL OF EDUCATION, LUNCH AND TEST PREP-
RATION COURSE. 
'''
# ===================================================================================================>


# ========================= ALL THE REQUIRED LIBRARIES ARE AVAILABLE HERE ===========================>
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# ===================================================================================================>


# ============================= DATA AND RELATED INFO IS AVAILABLE HERE =============================>
# DATA SET IS HERE ----------------------------------------->
df = pd.read_csv("DATA/stud.csv")
print("============ ALL THE COLUMNS OF THE DATASET =================================>")
print(df.columns)
print("=============================================================================>\n")
# print(df.shape)

# DATA RELATED INFO IS HERE -------------------------------->
'''
1. GENDER --> (MALE/FEMALE)
2. RACE/ETHNICITY -->(GROUP:A,B,C,D,E)
3. PARENTAL LEVEL OF EDUCATION --> (BACHELOR'S DEGREE, SOME COLLEGE, MASTER'S DEGREE, ASSOCIATE'S
 DEGREE, HIGH SCHOOL)
4. LUNCH --> HAVING LUNCH BEFORE TEST (STANDARD OR FREE/REDUCED)
5. TEST PREPARATION SCORE --> COMPLETE OR NOT COMPLETE BEFORE TEST
6. MATH SCORE
7. READING SCORE
8. WRITING SCORE
'''

# TO SHOW ALL THE ROWS ------------------------------------->
# pd.set_option('display.max_rows', None)

# TO SHOW ALL THE COLUMNS ---------------------------------->
# pd.set_option('display.max_columns', None)
# ===================================================================================================>


# =============== EXPLORATORY DATA ANALYSIS BEFORE FEATURE ENGINEERING IS DONE HERE =================>
# DATA CHECKS TO PERFORM ----------------------------------->
'''print(df.isnull().sum())# MISSING VALUE
print(df.duplicated().sum())# DUPLICATE VALUE
print(df.dtypes) # DATA TYPE
print(df.nunique().sum())# NUMBER UNIQUE VALUE'''

# CHECK STATISTICS OF DATASET ------------------------------>
'''print(df.describe())
   print(df[df['math_score']==0])'''

# EXPLORING ALL THE POSSIBLE CATEGORIES IN OUR DATASET ----->
'''
print("============ ALL THE POSSIBLE CATEGORIES IN THE DATASET =====================>")
print("CATEGORIES IN 'GENDER' VARIABLE : ", end=' ')
print(df['gender'].unique(), '\n')

print("CATEGORIES IN 'race_ethnicity' VARIABLE : ", end=' ')
print(df['race_ethnicity'].unique(), '\n')

print("CATEGORIES IN 'parental_level_of_education' VARIABLE : ", end=' ')
print(df['parental_level_of_education'].unique(), '\n')

print("CATEGORIES IN 'lunch' VARIABLE : ", end=' ')
print(df['lunch'].unique(), '\n')

print("CATEGORIES IN 'test_preparation_course' VARIABLE : ", end=' ')
print(df['test_preparation_course'].unique())
print("=============================================================================>")
'''
# ===================================================================================================>


# ================================== FEATURE ENGINEERING IS DONE HERE ===============================>
# SEPARATE THE CATEGORICAL AND THE NUMERICAL VALUES -------->
# NUMERICAL
numeric_features = [features for features in df.columns if df[features].dtype != 'O']
# print('WE HAVE {} NUMERICAL FEATURES: {}'.format(len(numeric_features), numeric_features))

# CATEGORICAL
categorical_features = [features for features in df.columns if df[features].dtype == 'O']
# print('WE HAVE {} CATEGORICAL FEATURES: {}'.format(len(categorical_features), categorical_features))

# ADDING TWO VALUES TOTAL SCORE AND AVERAGE IN THE DATASET ->
df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
df['average'] = df['total_score']/3
# print(df.head())

# STUDENTS WITH FULL MARKS IN READING, WRITING AND MATHS -->
reading_full = df[df['reading_score']==100]['average'].count()
writing_full = df[df['writing_score']==100]['average'].count()
maths_full = df[df['math_score']==100]['average'].count()

'''print("======= NUMBER OF STUDENTS WHO SCORED OUT OF IN THESE THREE FORMATS =========>")
print(f"NUMBER OF STUDENTS SCORE OUT OF IN READING: [{reading_full}]")
print(f"NUMBER OF STUDENTS SCORE OUT OF IN WRITING: [{writing_full}]")
print(f"NUMBER OF STUDENTS SCORE OUT OF IN MATHS: [{maths_full}]")
print("=============================================================================>\n")'''
# ===================================================================================================>


# ======================= VISUALIZATION OF THE DATA (EXPLORATORY DATA ANALYSIS) =====================>
# HISTOGRAM AND KDE PLOT ---------------------------------->
# AVERAGE ===========================>
'''fig, axs = plt.subplots(1,2,figsize = (13,5))
plt.subplot(121)
sns.histplot(data=df, x = 'average', bins=30,kde=True, color='r')
plt.subplot(122)
sns.histplot(data=df, x = 'average', bins=30,kde=True, hue='gender')

INSIGHT:
1--> FEMALE STUDENTS TENDS TO PERFORM WELL AS COMPARE TO MALE
'''

# TOTAL SCORE =======================>
'''fig, axs = plt.subplots(1,2,figsize = (13,5))
plt.subplot(121)
sns.histplot(data=df, x = 'total_score', bins=30,kde=True, color='r')
plt.subplot(122)
sns.histplot(data=df, x = 'total_score', bins=30,kde=True, hue='gender')

INSIGHT:
1--> FEMALE STUDENTS TENDS TO PERFORM WELL AS COMPARE TO MALE
'''

# LUNCH =============================>
'''plt.subplots(1,3, figsize=(20,6))
plt.subplot(131)
sns.histplot(df, x='average',kde=True, hue='lunch')
plt.subplot(132)
sns.histplot(df[df.gender=='female'], x='average',kde=True, hue='lunch')
plt.subplot(133)
sns.histplot(df[df.gender=='male'], x='average',kde=True, hue='lunch')

INSIGHT:
1--> STANDARD LUNCH HELPS PERFORM WELL IN EXAM
2--> BE IT MALE OR FEMALE BOTH
'''

# PARENTAL LEVEL OF EDUCATION =======>
# plt.subplot(1,3, figsize=(20,6))
# ===================================================================================================>
plt.show()
import pandas as pd
import os
print('current dir', os.getcwd())