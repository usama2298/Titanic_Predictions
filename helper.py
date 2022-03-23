from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, precision_recall_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

color = {"Magenta 100": "2A0A16", "Magenta 90": "57002B", "Magenta 80": "760A3A", "Magenta 70": "A11950",
         "Magenta 60": "D12765", "Magenta 50": "EE538B", "Magenta 40": "FA75A6", "Magenta 30": "FFA0C2",
         "Magenta 20": "FFCFE1", "Magenta 10": "FFF0F6", "Purple 100": "1E1033", "Purple 90": "38146B",
         "Purple 80": "4F2196", "Purple 70": "6E32C9", "Purple 60": "8A3FFC", "Purple 50": "A66EFA",
         "Purple 40": "BB8EFF", "Purple 30": "D0B0FF", "Purple 20": "E6D6FF", "Purple 10": "F7F1FF",
         "Blue 100": "051243", "Blue 90": "061F80", "Blue 80": "0530AD", "Blue 70": "054ADA", "Blue 60": "0062FF",
         "Blue 50": "408BFC", "Blue 40": "6EA6FF", "Blue 30": "97C1FF", "Blue 20": "C9DEFF", "Blue 10": "EDF4FF",
         "Teal 100": "081A1C", "Teal 90": "003137", "Teal 80": "004548", "Teal 70": "006161", "Teal 60": "007D79",
         "Teal 50": "009C98", "Teal 40": "00BAB6", "Teal 30": "20D5D2", "Teal 20": "92EEEE", "Teal 10": "DBFBFB",
         "Gray 100": "171717", "Gray 90": "282828", "Gray 80": "3D3D3D", "Gray 70": "565656", "Gray 60": "6F6F6F",
         "Gray 50": "8C8C8C", "Gray 40": "A4A4A4", "Gray 30": "BEBEBE", "Gray 20": "DCDCDC", "Gray 10": "F3F3F3"
         }
colors = []
colornum = 60
for i in [f'Blue {colornum}', f'Teal {colornum}', f'Magenta {colornum}', f'Purple {colornum}', f'Gray {colornum}']:
    colors.append(f'#{color[i]}')
palette = sns.color_palette(colors)


def ClassResult(estimator, xTrain, yTrain, xTest, yTest):
    estimator.fit(xTrain, yTrain)
    yPred = estimator.predict(xTest)

    print(classification_report(yTest, yPred))
    score_df = pd.DataFrame({'accuracy': accuracy_score(yTest, yPred),
                             'precision': precision_score(yTest, yPred),
                             'recall': recall_score(yTest, yPred),
                             'f1_score': f1_score(yTest, yPred),
                             'auc': roc_auc_score(yTest, yPred)},
                            index=pd.Index([0]))

    print(score_df)

    _, ax = plt.subplots(figsize=(5, 4))
    ax = sns.heatmap(confusion_matrix(yTest, yPred), annot=True,
                     cmap=colors, fmt='d', annot_kws={"size": 40, "weight": "bold"})
    labels = ['False', 'True']
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_yticklabels(labels[::-1], fontsize=15)
    ax.set_ylabel('Prediction', fontsize=20)
    ax.set_xlabel('Actual', fontsize=20)


def ROC_PRcurve(model, xTest, yTest):

    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(16, 8)
    yProb = model.predict_proba(xTest)

    ax = axes[0]
    fpr, tpr, _ = roc_curve(yTest, yProb[:, 1])
    ax.plot(fpr, tpr, color=colors[0], linewidth=5)
    ax.plot([0, 1], [0, 1], ls='--', color='black', lw=.3)
    ax.set(xlabel='False Positive Rate',
           ylabel='True Positive Rate',
           xlim=[-.01, 1.01], ylim=[-.01, 1.01],
           title='ROC curve')
    ax.grid(True)

    ax = axes[1]
    precision, recall, _ = precision_recall_curve(yTest, yProb[:, 1])
    ax.plot(recall, precision, color=colors[1], linewidth=5)
    ax.set(xlabel='Recall', ylabel='Precision',
           xlim=[-.01, 1.01], ylim=[-.01, 1.01],
           title='Precision-Recall curve')
    ax.grid(True)

    plt.tight_layout()
