
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocessing import load_data, preprocess_data
from features import create_features


def train_model(filepath):
    df=load_data(filepath)
    df=create_features(df)
    df=preprocess_data(df)
    y = df["Survived"]   
    X = df.drop(columns=["Survived"])
    X_train,X_val,y_train,y_val=train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_val)
    val_accuracy= accuracy_score(y_val,y_pred)
    print("Validation Accuracy:",val_accuracy)
    joblib.dump(model,"outputs/logistic_regression_model.pkl")
    return model 

if __name__=="__main__":
    train_model("data/train.csv")
