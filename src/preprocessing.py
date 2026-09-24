import pandas as pd
from features import create_features

def load_data(filepath):
    return pd.read_csv(filepath)

def fill_missing_embarked(df):
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df

def fill_missing_age_by_title(df):
    title_age_medians = df.groupby("Title")["Age"].median()
    df["Age"] = df.apply(
        lambda x: title_age_medians[x["Title"]]
        if pd.isnull(x["Age"])
            else x["Age"],
            axis = 1 
    )
    return df

def fill_missing_fare(df):
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    return df

def encode_sex(df):
    df["Sex"] = df["Sex"].map({
        "male":1 ,
        "female":0
    })
    return df

def encode_embarked(df) :
    df=pd.get_dummies(
        df,
        columns=["Embarked"],
        dtype = int
        )
    return df 

def drop_unnecessary_columns(df):
    df= df.drop(
        columns = [
            "Name" , "Ticket","Cabin","PassengerId"
        ]
    )
    return df
def encode_title(df):
    df = pd.get_dummies(
        df,
        columns=["Title"],
        dtype = int
    )
    return df

def preprocess_data(df):

    df=fill_missing_embarked(df)
    df=fill_missing_age_by_title(df)
    df=encode_sex(df)
    df=encode_embarked(df)
    df=encode_title(df)
    df = drop_unnecessary_columns(df)
    return df


def prepare_data(df):
    df = fill_missing_fare(df)
    df= create_features(df)
    df=preprocess_data(df)
    return df