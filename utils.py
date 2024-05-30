
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, recall_score, f1_score, brier_score_loss, roc_auc_score

def plot_roc_curve(model, X_test, y_test, modelName):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{modelName} ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

def calculate_default_rate(y_test, y_pred_proba):
    df = pd.DataFrame({'True': y_test, 'Proba': y_pred_proba})
    df['Quintil'] = pd.qcut(df['Proba'], 5, labels=False, duplicates='drop')
    default_rate = df[df['True'] == 0].groupby('Quintil').size() / df.groupby('Quintil').size()
    return default_rate.fillna(0)

def default_rate_by_quintile(model, X_test, y_test, modelName):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    default_rate = calculate_default_rate(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.bar(np.arange(len(default_rate)), default_rate, color='darkorange', edgecolor='grey', label=modelName)

    plt.xlabel('Quintile', fontweight='bold')
    plt.ylabel('Default Rate')
    plt.xticks(np.arange(len(default_rate)), range(1, 6))
    plt.title('Default Rate by Quintiles')
    plt.legend()
    plt.show()


def calculate_metrics(pipeline, X_test, y_test):
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred_binary = pipeline.predict(X_test)

    avg_precision = average_precision_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    metrics_df = pd.DataFrame({
        'Metric': ['Average Precision Score', 'Recall', 'F1 Score', 'AUC', 'Brier Score'],
        'Score': [avg_precision, recall, f1, auc, brier]
    })

    return metrics_df


