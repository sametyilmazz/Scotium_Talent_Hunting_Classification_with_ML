##############################################################
#### Scouting Classification with Machine Learning  ####

######################## Business Problem #########################

# Given the characteristics of football players watched by Scouts
# Predicting which class (average, highlighted) players are based on points.

#################### Dataset Story ######################

# The data set consists of information from Scoutium,which includes the features
# and scores of the football players evaluated by the scouts according to
# the characteristics of the football players observed in the matches.


# Total Observation 10.730

# task_response_id  : The set of a scout's assessments of all players on a team's roster in a match
# match_id          : The id of the relevant match
# evaluator_id      : The id of the scout
# player_id         : The id of the relevant player
# position_id       : The id of the position played by the relevant player in that match
# 1 : Goalkeeper
# 2 : Stopper
# 3 : Right Back
# 4 : Left Back
# 5 : Defensive Midfielder
# 6 : Central Midfielder
# 7 : Right Wing
# 8 : Left Wing
# 9 : OffensiveMidfielder
# 10: Striker
# analysis_id       : A set containing a scout's attribute evaluations of a player in a match
# attribute_id      : The id of each attribute the players were evaluated on
# attribute_value   : Value (points) given by a scout to a player's attribute


# Total Observation 322

# task_response_id  : The set of a scout's assessments of all players on a team's roster in a match
# match_id          : The id of the relevant match
# evaluator_id      : The id of the scout
# player_id         : The id of the relevant player
# potential_label   : Label indicating the final decision of a scout regarding a player in a match. (target variable)


##### Importing the Essential Libraries, Metrics

import itertools
import warnings
import pandas as pd
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
warnings.simplefilter(action='ignore', category=Warning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Loading the Data

sc_attributes = pd.read_csv("pythonProject/DATASETS/scoutium_attributes.csv", sep=";")
sc_potential_labels = pd.read_csv("pythonProject/DATASETS/scoutium_potential_labels.csv", sep=";")

df_sc = pd.merge(sc_attributes, sc_potential_labels, how='left',
                 on=["task_response_id", "match_id", "evaluator_id", "player_id"])


###### 1- EDA (Exploratory Data Analysis) ######

######### SUMMARY #########

def summarize(dataframe):
    print(f"Dataset Shape: {dataframe.shape}")
    summary = pd.DataFrame(dataframe.dtypes, columns=["dtypes"])
    summary = summary.reset_index()
    summary["Name"] = summary["index"]
    summary = summary[["Name", "dtypes"]]
    summary["Missing"] = dataframe.isnull().sum().values
    summary["Uniques"] = dataframe.nunique().values
    summary["First Value"] = dataframe.loc[0].values
    summary["Second Value"] = dataframe.loc[1].values
    summary["Third Value"] = dataframe.loc[2].values
    summary["Fourth Value"] = dataframe.loc[3].values
    summary["Fifth Value"] = dataframe.loc[4].values

    return summary

summarize(df_sc)

# Dataset Shape: (10730, 9)
# Out[4]:
#                Name   dtypes  Missing  Uniques First Value Second Value Third Value Fourth Value Fifth Value
# 0  task_response_id    int64        0       29        4915         4915        4915         4915        4915
# 1          match_id    int64        0       17       62935        62935       62935        62935       62935
# 2      evaluator_id    int64        0       10      177676       177676      177676       177676      177676
# 3         player_id    int64        0      277     1361061      1361061      177676       177676      177676
# 4       position_id    int64        0       10           2            2           2            2           2
# 5       analysis_id    int64        0      323    12818495     12818495    12818495     12818495    12818495
# 6      attribute_id    int64        0       39        4322         4323        4324         4325        4326
# 7   attribute_value  float64        0       10      56.000       56.000      67.000       56.000      45.000
# 8   potential_label   object        0        3     average      average     average      average     average


# Let's remove the Keeper (1) class in position_id from the dataset.
df_sc["position_id"].value_counts()
# 2     1972
# 6     1428
# 10    1088
# 8     1020
# 7      986
# 3      986
# 4      884
# 9      850
# 5      816
# 1      700

df_sc = df_sc[df_sc.position_id != 1]

# Let's remove the below_average class in potential_label from the dataset.
df_sc["potential_label"].value_counts()
# average          7922
# highlighted      1972
# below_average     136

df_sc = df_sc[df_sc.potential_label != "below_average"]

# Let's implement the pivot_table method with one player per row.

df_sc_pivot = df_sc.pivot_table(index=["player_id", "position_id", "potential_label"],
                  columns="attribute_id", values="attribute_value").reset_index()
df_sc_pivot.columns = df_sc_pivot.columns.astype(str)
df_sc_pivot.head()
# attribute_id  player_id  position_id potential_label  4322  4323  4324  4325  4326  4327  4328  4329  4330  4332  4333  4335  4338  4339  4340  4341  4342  4343  4344  4345  4348  4349  4350  4351  4352  4353  4354  4355  4356  4357  4407  4408  4423  4426
# 0               1355710            7         average  50.5  50.5  34.0  50.5  45.0  45.0  45.0  45.0  50.5  56.0  39.5  34.0  39.5  39.5  45.0  45.0  50.5  28.5  23.0  39.5  28.5  28.5  45.0  50.5  56.0  34.0  39.5  50.5  34.0  34.0  56.0  34.0  34.0  56.0
# 1               1356362            9         average  67.0  67.0  67.0  67.0  67.0  67.0  67.0  67.0  67.0  67.0  67.0  67.0  67.0  67.0  67.0  67.0  67.0  67.0  56.0  67.0  67.0  56.0  67.0  67.0  67.0  67.0  78.0  67.0  67.0  67.0  67.0  67.0  56.0  78.0
# 2               1356375            3         average  67.0  67.0  67.0  67.0  67.0  67.0  67.0  78.0  67.0  67.0  78.0  56.0  67.0  67.0  67.0  67.0  67.0  56.0  56.0  67.0  67.0  56.0  56.0  67.0  67.0  67.0  78.0  67.0  67.0  67.0  67.0  67.0  56.0  78.0
# 3               1356375            4         average  67.0  78.0  67.0  67.0  67.0  78.0  78.0  78.0  56.0  67.0  67.0  67.0  78.0  78.0  56.0  67.0  67.0  45.0  45.0  56.0  67.0  67.0  67.0  67.0  78.0  67.0  67.0  67.0  56.0  67.0  56.0  67.0  45.0  56.0
# 4               1356411            9         average  67.0  67.0  78.0  78.0  67.0  67.0  67.0  67.0  89.0  78.0  67.0  67.0  67.0  56.0  56.0  67.0  78.0  56.0  56.0  67.0  56.0  67.0  56.0  67.0  67.0  56.0  67.0  67.0  56.0  67.0  89.0  56.0  67.0  78.0


# Label Encoder
labelencoder = LabelEncoder()
df_sc_pivot["potential_label"] = labelencoder.fit_transform(df_sc_pivot["potential_label"])
df_sc_pivot.head()

# attribute_id  player_id  position_id  potential_label   4322   4323   4324   4325   4326   4327   4328   4329   4330   4332   4333   4335   4338   4339   4340   4341   4342   4343   4344   4345   4348   4349   4350   4351   4352   4353   4354   4355   4356   4357   4407   4408   4423   4426
# 0               1355710            7                0 50.500 50.500 34.000 50.500 45.000 45.000 45.000 45.000 50.500 56.000 39.500 34.000 39.500 39.500 45.000 45.000 50.500 28.500 23.000 39.500 28.500 28.500 45.000 50.500 56.000 34.000 39.500 50.500 34.000 34.000 56.000 34.000 34.000 56.000
# 1               1356362            9                0 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 56.000 67.000 67.000 56.000 67.000 67.000 67.000 67.000 78.000 67.000 67.000 67.000 67.000 67.000 56.000 78.000
# 2               1356375            3                0 67.000 67.000 67.000 67.000 67.000 67.000 67.000 78.000 67.000 67.000 78.000 56.000 67.000 67.000 67.000 67.000 67.000 56.000 56.000 67.000 67.000 56.000 56.000 67.000 67.000 67.000 78.000 67.000 67.000 67.000 67.000 67.000 56.000 78.000
# 3               1356375            4                0 67.000 78.000 67.000 67.000 67.000 78.000 78.000 78.000 56.000 67.000 67.000 67.000 78.000 78.000 56.000 67.000 67.000 45.000 45.000 56.000 67.000 67.000 67.000 67.000 78.000 67.000 67.000 67.000 56.000 67.000 56.000 67.000 45.000 56.000
# 4               1356411            9                0 67.000 67.000 78.000 78.000 67.000 67.000 67.000 67.000 89.000 78.000 67.000 67.000 67.000 56.000 56.000 67.000 78.000 56.000 56.000 67.000 56.000 67.000 56.000 67.000 67.000 56.000 67.000 67.000 56.000 67.000 89.000 56.000 67.000 78.000

# Let's assign the numeric variable columns to a list with the name "num_cols".
num_cols = [col for col in df_sc_pivot.columns if col not in
            ["player_id", "position_id", "potential_label"]]

df_sc_pivot_num = df_sc_pivot[num_cols]
df_sc_pivot_num.head()
# attribute_id   4322   4323   4324   4325   4326   4327   4328   4329   4330   4332   4333   4335   4338   4339   4340   4341   4342   4343   4344   4345   4348   4349   4350   4351   4352   4353   4354   4355   4356   4357   4407   4408   4423   4426
# 0            50.500 50.500 34.000 50.500 45.000 45.000 45.000 45.000 50.500 56.000 39.500 34.000 39.500 39.500 45.000 45.000 50.500 28.500 23.000 39.500 28.500 28.500 45.000 50.500 56.000 34.000 39.500 50.500 34.000 34.000 56.000 34.000 34.000 56.000
# 1            67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 67.000 56.000 67.000 67.000 56.000 67.000 67.000 67.000 67.000 78.000 67.000 67.000 67.000 67.000 67.000 56.000 78.000
# 2            67.000 67.000 67.000 67.000 67.000 67.000 67.000 78.000 67.000 67.000 78.000 56.000 67.000 67.000 67.000 67.000 67.000 56.000 56.000 67.000 67.000 56.000 56.000 67.000 67.000 67.000 78.000 67.000 67.000 67.000 67.000 67.000 56.000 78.000
# 3            67.000 78.000 67.000 67.000 67.000 78.000 78.000 78.000 56.000 67.000 67.000 67.000 78.000 78.000 56.000 67.000 67.000 45.000 45.000 56.000 67.000 67.000 67.000 67.000 78.000 67.000 67.000 67.000 56.000 67.000 56.000 67.000 45.000 56.000
# 4            67.000 67.000 78.000 78.000 67.000 67.000 67.000 67.000 89.000 78.000 67.000 67.000 67.000 56.000 56.000 67.000 78.000 56.000 56.000 67.000 56.000 67.000 56.000 67.000 67.000 56.000 67.000 67.000 56.000 67.000 89.000 56.000 67.000 78.000

# StandardScaler
sc = StandardScaler()
sc.fit(df_sc_pivot[num_cols])
df_sc_pivot[num_cols] = sc.transform(df_sc_pivot[num_cols])
df_sc_pivot[num_cols].head()

# attribute_id   4322   4323   4324   4325   4326   4327   4328   4329   4330   4332   4333   4335   4338   4339   4340   4341   4342   4343   4344   4345   4348   4349   4350   4351  4352   4353   4354   4355   4356   4357   4407   4408   4423   4426
# 0            -0.543 -0.559 -1.405 -0.438 -0.767 -0.795 -0.907 -0.792 -0.446 -0.123 -1.224 -1.036 -1.126 -1.009 -0.542 -0.690 -0.535 -1.067 -1.206 -1.005 -1.314 -1.042 -0.693 -0.436 0.013 -1.282 -1.132 -0.495 -1.235 -1.520 -0.143 -1.487 -0.955 -0.253
# 1             0.595  0.561  0.679  0.683  0.723  0.723  0.601  0.404  0.728  0.691  0.459  0.996  0.632  0.840  0.912  0.799  0.588  1.230  0.750  0.742  0.879  0.670  0.848  0.717 0.787  0.814  1.053  0.632  0.915  0.768  0.530  0.669  0.404  1.042
# 2             0.595  0.561  0.679  0.683  0.723  0.723  0.601  1.002  0.728  0.691  1.132  0.319  0.632  0.840  0.912  0.799  0.588  0.574  0.750  0.742  0.879  0.670  0.077  0.717 0.787  0.814  1.053  0.632  0.915  0.768  0.530  0.669  0.404  1.042
# 3             0.595  1.308  0.679  0.683  0.723  1.482  1.355  1.002 -0.055  0.691  0.459  0.996  1.335  1.580  0.185  0.799  0.588 -0.083  0.098  0.043  0.879  1.355  0.848  0.717 1.560  0.814  0.428  0.632  0.198  0.768 -0.143  0.669 -0.276 -0.253
# 4             0.595  0.561  1.373  1.430  0.723  0.723  0.601  0.404  2.294  1.505  0.459  0.996  0.632  0.101  0.185  0.799  1.337  0.574  0.750  0.742  0.252  1.355  0.077  0.717 0.787  0.115  0.428  0.632  0.198  0.768  1.874 -0.050  1.083  1.042

# Modelling

y = df_sc_pivot["potential_label"]
X = df_sc_pivot.drop(["potential_label", "player_id"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

def base_models(X, y, scoring):
    print("Base Models....")
    models = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))]

    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=10, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


scores = ["roc_auc", "f1", "precision", "recall", "accuracy"]

for i in scores:
    base_models(X, y, i)

# Base Models....
# roc_auc: 0.8453 (LR)
# roc_auc: 0.7257 (KNN)
# roc_auc: 0.8439 (SVC)
# roc_auc: 0.7335 (CART)
# roc_auc: 0.9127 (RF)
# roc_auc: 0.8727 (Adaboost)
# roc_auc: 0.8685 (GBM)
# roc_auc: 0.8558 (XGBoost)
# roc_auc: 0.8982 (LightGBM)
# roc_auc: 0.9041 (CatBoost)

# Base Models....
# f1: 0.5685 (LR)
# f1: 0.4279 (KNN)
# f1: 0.0333 (SVC)
# f1: 0.5244 (CART)
# f1: 0.6044 (RF)
# f1: 0.5999 (Adaboost)
# f1: 0.6171 (GBM)
# f1: 0.611 (XGBoost)
# f1: 0.6633 (LightGBM)
# f1: 0.5938 (CatBoost)

# Base Models....
# precision: 0.7738 (LR)
# precision: 0.775 (KNN)
# precision: 0.1 (SVC)
# precision: 0.5571 (CART)
# precision: 0.9167 (RF)
# precision: 0.7971 (Adaboost)
# precision: 0.735 (GBM)
# precision: 0.7398 (XGBoost)
# precision: 0.8071 (LightGBM)
# precision: 0.93 (CatBoost)

# Base Models....
# recall: 0.49 (LR)
# recall: 0.31 (KNN)
# recall: 0.02 (SVC)
# recall: 0.5867 (CART)
# recall: 0.4867 (RF)
# recall: 0.5433 (Adaboost)
# recall: 0.5067 (GBM)
# recall: 0.5767 (XGBoost)
# recall: 0.5933 (LightGBM)
# recall: 0.47 (CatBoost)

# Base Models....
# accuracy: 0.8525 (LR)
# accuracy: 0.845 (KNN)
# accuracy: 0.7971 (SVC)
# accuracy: 0.8008 (CART)
# accuracy: 0.8893 (RF)
# accuracy: 0.8561 (Adaboost)
# accuracy: 0.8561 (GBM)
# accuracy: 0.8563 (XGBoost)
# accuracy: 0.8817 (LightGBM)
# accuracy: 0.8817 (CatBoost)


# Feature Importance
def feature_importance(model, X, y):
    model.fit(X_train, y_train)
    feature_importance = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=["importance"])
    feature_importance.sort_values(by="importance", ascending=False, inplace=True)
    plt.figure(figsize=(12, 12))
    sns.barplot(x=feature_importance.importance, y=feature_importance.index)
    plt.title(model)
    plt.show(block=True)

feature_importance(RandomForestClassifier(), X, y)

feature_importance(LGBMClassifier(), X, y)

feature_importance(DecisionTreeClassifier(), X, y)

feature_importance(GradientBoostingClassifier(), X, y)
