import pandas as pd 

def extract_title(name):
    return name.split(",")[1].split(".")[0].strip()


def create_title(df):

    df["Title"] = df["Name"].apply(extract_title)

    df["Title"] = df["Title"].apply(
        lambda x: x if x in ["Mr","Miss","Mrs","Master"] else "Rare"
    )
    return df


def create_family_size(df) :

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    return df 

def create_is_alone(df):
    df["IsAlone"]=df["FamilySize"].apply(
        lambda x: 1 if x ==1 else 0
    )
    return df 

def create_has_cabin(df):
    df["HasCabin"] = df["Cabin"].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )
    return df

def create_fare_per_person(df):
    df["FareperPerson"]=df["Fare"]/df["FamilySize"]
    return df


def create_features(df):
    df = create_title(df)
    df = create_family_size(df)
    df= create_fare_per_person(df)
    df = create_is_alone(df)
    df = create_has_cabin(df)
    return df