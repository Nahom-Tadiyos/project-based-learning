
import pandas as pd
import numpy as np

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

    #Convert Gender
    df["Sex"] = df["Sex"].map({'male':1, "female":0})

    #Feature enginerring
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df['FamilySize'] == 0, 1, 0)
    df["FareBin"] = pd.qcut()(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0,12,20,40,60, np.inf], labels = False)

    return df
