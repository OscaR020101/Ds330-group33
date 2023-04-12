from sklearn.metrics import confusion_matrix
import numpy as np
def ConfusionMatrix(y_true, y_pred):
    # assume y_true and y_pred are the true labels and predicted labels, respectively
    cm = confusion_matrix(y_true, y_pred)

    # calculate accuracy
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)

    # calculate recall
    recall = cm[1,1] / (cm[1,1] + cm[1,0])

    # calculate precision
    precision = cm[1,1] / (cm[1,1] + cm[0,1])

    # calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Confusion Matrix:\n", cm)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 Score:", f1_score)

