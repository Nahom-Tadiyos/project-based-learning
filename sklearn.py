
import pandas as pd
import numpy as np

from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.neighbors import *
from sklearn.metrics import *

from sklearn import *
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("imgs/tested.csv")
data.info()
print(data)

#data cleaning
def preprocces_data(df):
    df.drop(columns=["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"])

    df["Embarked"].fillna("S", inplace = True)
    df.drop(columns=["Embarked"], inplace = True)

    fill_missing_ages(df)

    #Convert Gender
    df["Sex"] = df["Sex"].map({'male':1, "female":0})

    #Feature enginerring
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df['FamilySize'] == 0, 1, 0)
    df["FareBin"] = pd.qcut()(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0,12,20,40,60, np.inf], labels = False)

    return df

def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
    
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row['Age']) else row["Age"],
                         axis = 1)
    
data = preprocces_data(data)

#Create Features / Target Variables (Make Flashcards)
X = data.drop(columns=["Survived"])
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 

#ML Preprocessing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

