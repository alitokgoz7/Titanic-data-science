import joblib
import pandas as pd 
from preprocessing import load_data, prepare_data

def predict_model(test_filepath,model_filepath) :
    df = load_data(test_filepath)
    passenger_id=df["PassengerId"]
    df = prepare_data(df)
    model=joblib.load(model_filepath)
    X_test = df 
    predictions = model.predict(X_test)
    submission = pd.DataFrame({
        "PassengerId" : passenger_id,
        "Survived": predictions
    })
    submission.to_csv("outputs/submission.csv",index=False)
    return predictions

if __name__ == "__main__":
    predict_model("data/test.csv", "outputs/logistic_regression_model.pkl")