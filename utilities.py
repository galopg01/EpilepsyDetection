from matplotlib.pylab import plt
from numpy import arange
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt
import os

def plot_losses(epochs, history, architecture, patch_size, k, N):
    # Plot and label the training and validation loss values
    plt.plot(epochs, history["train_loss"], label='Training Loss')
    plt.plot(epochs, history["val_loss"], label='Validation Loss')

    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    total_epochs = len(list(epochs)) + 1
    # Set the tick locations
    plt.xticks(arange(0, total_epochs, 2))

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("graphics/" + architecture + "/x" + str(patch_size) + "/" + str(N) +  "/" + architecture + "_x" + str(patch_size) + "_k_" + str(k) + "_loss.png")
    plt.show()
    
def plot_accuracy(epochs, history, architecture, patch_size, k, N):
    # Plot and label the training and validation accuracy values
    plt.plot(epochs, history["train_acc"], label='Training Accuracy')
    plt.plot(epochs, history["val_acc"], label='Validation Accuracy')

    # Add in a title and axes labels
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    total_epochs = len(list(epochs)) + 1
    # Set the tick locations
    plt.xticks(arange(0, total_epochs, 2))

    # Display the plot
    plt.legend(loc='best')
    plt.savefig("graphics/" + architecture + "/x" + str(patch_size) + "/" + str(N) +  "/" + architecture + "_x" + str(patch_size) + "_k_" + str(k) + "_acc.png")
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, fold, ax):
    ax.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for Fold ' + str(fold))
    ax.legend(loc="lower right")

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names, fold, ax):
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g', ax=ax)
    ax.set_title('Confusion Matrix for Fold ' + str(fold))
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

def plot_confusion_matrices(confusion_matrices, architecture, patch_size, k, N):
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    for i, cm in enumerate(confusion_matrices):
        class_names = ['nonLesion', 'lesion']
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
        # Accessing individual axes objects
        sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g', ax=axs.flat[i])  # Use axs.flat[i] to access individual axes
        axs.flat[i].set_title(f'Fold {i} Confusion Matrix')
        axs.flat[i].set_xlabel('Predicted labels')
        axs.flat[i].set_ylabel('True labels')
    
    # Hide unused subplots
    for ax in axs.flat[len(confusion_matrices):]:
        ax.axis('off')
    
    plt.suptitle('Confusion matrices for ' + architecture, fontsize=16)
    
    
    plt.tight_layout()
    plt.savefig("graphics/" + architecture + "/x" + str(patch_size) + "/" + str(N) + "/" + architecture + "_x" + str(patch_size) + "_k_" + str(k) + "_matrices.png")
    plt.show()

def show_roc_curves(roc_curves, architecture, patch_size, k, N):
    plt.figure(figsize=(10, 5))

    for i, roc_curve_data in enumerate(roc_curves):
        fpr, tpr, roc_auc = roc_curve_data
        plt.plot(fpr, tpr, lw=2, label=f'Fold {i} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("graphics/" + architecture + "/x" + str(patch_size) + "/" + str(N) + "/" + architecture + "_x" + str(patch_size) + "_k_" + str(k) + "_curves.png")
    plt.savefig("graphics/" + architecture + "/x" + str(patch_size) + "/" + str(N) + "/" + architecture + "_x" + str(patch_size) + "_k_" + str(k) + "_curves.pdf")
    plt.show()

def show_confusion_matrices(confusion_matrices, architecture, patch_size, k, N):
    num_rows = k // 3 
    num_cols = 3 
    
    if k % 3 != 0:  
        num_rows += 1
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))

    for i, cm in enumerate(confusion_matrices):
        class_names = ['nonLesion', 'lesion']
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
        # Accessing individual axes objects
        sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g', ax=axs.flat[i])  # Use axs.flat[i] to access individual axes
        axs.flat[i].set_title(f'Fold {i} Confusion Matrix')
        axs.flat[i].set_xlabel('Predicted labels')
        axs.flat[i].set_ylabel('True labels')
    
    # Hide unused subplots
    for ax in axs.flat[len(confusion_matrices):]:
        ax.axis('off')
    
    # Add legend
    #handles = [plt.Rectangle((0,0),1,1,fc='w', edgecolor = 'none'), plt.Rectangle((0,0),1,1,fc='w', edgecolor = 'none')]
    
    plt.suptitle('Confusion matrices for ' + architecture, fontsize=16)
    
    
    plt.tight_layout()
    plt.savefig("graphics/" + architecture + "/x" + str(patch_size) + "/" + str(N) + "/" + architecture + "_x" + str(patch_size) + "_k_" + str(k) + "_matrices.png")
    plt.savefig("graphics/" + architecture + "/x" + str(patch_size) + "/" + str(N) + "/" + architecture + "_x" + str(patch_size) + "_k_" + str(k) + "_matrices.pdf")
    plt.show()
        

def show_metrics():

    modelos = {}
    tamanos = ['32', '48', '64', '96', '128']
    tipos = ['', 'noSharpen', 'sharpen']

    data = []

    for tamaño in tamanos:
        for tipo in tipos:
            if tipo:
                tipo_str = f"-{tipo}"
            else:
                tipo_str = ""

            
            modelo = None
            if os.path.exists(f'./metrics/resultsN-{tamaño}{tipo_str}.json'):
                with open(f'./metrics/resultsN-{tamaño}{tipo_str}.json') as f:
                    modelo = json.load(f)
                    for n, metrics in modelo['alexnet'].items():
                        data.append({'N': int(n), 'F-score': metrics['F-score'], 'Tamaño': tamaño, 'Tipo': tipo})

    if modelo:
        grouped_data = {}
        for d in data:
            n = d['N']
            if n not in grouped_data:
                grouped_data[n] = []
            grouped_data[n].append(d['F-score'])

        
        plt.figure(figsize=(10, 6))

        boxplot_data = [grouped_data[n] for n in sorted(grouped_data.keys())]
        boxplot_labels = [f'N={n}' for n in sorted(grouped_data.keys())]

        plt.boxplot(boxplot_data, labels=boxplot_labels)

        plt.xlabel('N')
        plt.ylabel('F-score')
        plt.title('Desempeño de Alexnet en diferentes configuraciones')

        plt.grid(True)
        plt.show()
