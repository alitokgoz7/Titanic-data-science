from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,accuracy_score,roc_auc_score

def evaluate_model(y_val, y_pred, y_prob):
    accuracy = accuracy_score(y_val,y_pred)
    precision = precision_score(y_val,y_pred)
    recall = recall_score(y_val,y_pred)
    f1score = f1_score(y_val,y_pred)
    cm = confusion_matrix(y_val,y_pred)
    roc_auc = roc_auc_score(y_val,y_prob)
    return accuracy,precision,recall,f1score,cm,roc_auc
