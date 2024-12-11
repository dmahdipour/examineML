import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from pandas import DataFrame


class ModelResultVisualizer:
    def __init__(self):
        pass
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")   
        plt.title("Confusion Matrix")
        plt.show()
        
    def plot_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = DataFrame(report).transpose()
        plt.figure(figsize=(10, 6))
        sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="coolwarm", fmt=".2f", cbar=False)
        plt.title("Classification Report")
        plt.show()
        
    def plot_roc_curve(self, y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid()
        plt.show()
    
    def store_results(self, model_name, best_params, accuracy, results_file='results_summary.txt'):
        with open(results_file, 'a') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Best Parameters: {best_params}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write("\n--------------------------\n")