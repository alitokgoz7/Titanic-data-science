import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from preprocessing import load_data, prepare_data

from evaluate import evaluate_model

def train_model(filepath):
    df=load_data(filepath)
    df = prepare_data(df)
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
    y_prob =model.predict_proba(X_val)[:,1]
    accuracy,precision,recall,f1score,cm,roc_auc=evaluate_model(y_val,y_pred,y_prob)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1score)
    print("ROC-AUC:", roc_auc)
    print("Confusion Matrix:")
    print(cm)

    joblib.dump(model,"outputs/logistic_regression_model.pkl")
    return model 

if __name__=="__main__":
    train_model("data/train.csv")
