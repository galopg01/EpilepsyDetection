from sklearn.model_selection import KFold
import torchvision
from collections import Counter
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import random
from utilities import *
from dataset import *
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch.nn as nn
import torch
import torch.nn.functional as F
from patchMain import *
import argparse

# Configuration options

parser = argparse.ArgumentParser(description="Training script")

parser.add_argument('--architecture', type=str, default="alexnet", help='Model architecture')
parser.add_argument('--k', type=int, default=5, help='Folds for cross validation')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--N', type=int, default=1, help='Number of positive pixels')
parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
parser.add_argument('--mode', type=str, default="normal", help='Mode')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

args = parser.parse_args()

architecture = args.architecture
k_folds = args.k
num_epochs = args.epochs
N= args.N
patch_size= args.patch_size
mode= args.mode
lr = args.lr

sharpen = True if mode == "sharpen" else False

if len(os.listdir('./results/x' + str(patch_size) + '/' + str(N) + '/' + mode +'/test/lesion')) == 0:
    
    delete_images(patch_size,N,mode)
    if (mode == "sharpen" or mode == "noSharpen"):
        generate_3d_patches(patch_size, N, mode)
    else:
        generate_patches(patch_size, N, mode)

    delete_black_images(patch_size,N,mode)
    imgsToCsv(patch_size,N,mode)


# For fold results
results = {}
losses = {}
  
# Set fixed random number seed
torch.manual_seed(42)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }


stride= patch_size

if sharpen:
    train_data = EpilepsyDataset("","./testData/" + str(N) + "/" + mode + "/trainx" + str(patch_size) + ".csv",transform=transform_sharpen)
    test_data = EpilepsyDataset("","./testData/" + str(N) + "/" + mode + "/testx" + str(patch_size) + ".csv",transform=transform_sharpen)
else:  
    train_data = EpilepsyDataset("","./testData/" + str(N) + "/" + mode + "/trainx" + str(patch_size) + ".csv",transform=transform)
    test_data = EpilepsyDataset("","./testData/" + str(N) + "/" + mode + "/testx" + str(patch_size) + ".csv",transform=transform)

print("Aumentado de datos...")

aug = []

limit = 0.2
if (patch_size == 128):
    limit = 0.12
elif (patch_size == 96):
    limit = 0.08
elif (patch_size == 64):
    limit = 0.08
elif (patch_size == 48):
    limit = 0.05
else:
    limit = 0.05

for j, item in enumerate(train_data):
    if(item[2]==1 or random.random() < limit):
        aug.append(item)

if (patch_size == 48):
    
    if sharpen:
        data_new1 = EpilepsyDataset("","./testData/" + str(N) + "/" + mode + "trainx" + str(patch_size) + ".csv",transform=transform_sharpen)
    else:
        data_new1 = EpilepsyDataset("","./testData/" + str(N) + "/" + mode + "/trainx" + str(patch_size) + ".csv",transform=transform)

    for j, item in enumerate(data_new1):
        if(item[2]==1):
            aug.append(item)
   

train_data = aug

train_classes = [label for _, _, label in train_data]
print(Counter(i.item() for i in train_classes))

trainCsv = pd.read_csv("./testData/" + str(N) + "/" + mode + "/trainx" + str(patch_size) + ".csv")
p = trainCsv["subject"].unique()

confusion_matrices = []
roc_curves = []

test_ds = test_data
test_dl = DataLoader(test_ds, shuffle=False, batch_size=64)


for k, (train_ids, test_ids) in enumerate(kfold.split(p)):
    # Print
    print('FOLD {',k,'}')
    print('--------------------------------')
    print('train: ', train_ids, '   test:', test_ids, '\n')
                           
    print('Training...')

    ids_train = [idx for idx, elem in enumerate(train_data) if int(elem[1]) in p[train_ids]]
    ids_val = [idx for idx, elem in enumerate(train_data) if int(elem[1]) in p[test_ids]]

    train_ds = torch.utils.data.Subset(train_data, ids_train)
    val_ds = torch.utils.data.Subset(train_data, ids_val)
    
    
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=64)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #training with either cpu or cuda

    if (architecture=="alexnet"):
        model = torchvision.models.alexnet()
    elif (architecture == "mobilenet"):
        model = torchvision.models.mobilenet_v3_small()
    else:
        model = torchvision.models.shufflenet_v2_x1_5()
    
    model = model.to(device=device) #to send the model for training on either cuda or cpu

    ## Loss and optimizer
    learning_rate = lr #1e-4
    load_model = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= learning_rate) #Adam seems to be the most popular for deep learning

    for epoch in range(num_epochs): #I decided to train the model for 20 epochs
        loss_ep = 0
        loss_val= 0
        train_correct = 0
        train_samples = 0
        #gpu_usage()  

        for batch_idx, (data, sub, targets) in enumerate(train_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            ## Forward Pass
            optimizer.zero_grad()
            scores = model(data)
            targets = targets.long()
            loss = criterion(scores,targets)
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
            _, predictions = scores.max(1)
            train_correct += (predictions == targets).sum()
            train_samples += predictions.size(0)
        #print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}")

        with torch.no_grad():
            val_correct = 0
            val_samples = 0
            for batch_idx, (data, sub, targets) in enumerate(val_dl):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                scores = model(data)
                targets = targets.long()
                loss = criterion(scores,targets)
                loss_val += loss.item()
                _, predictions = scores.max(1)
                val_correct += (predictions == targets).sum()
                val_samples += predictions.size(0)
          #  print(
           #     f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
            #)
        print(f'Epoch [{epoch}] => train_loss: {loss_ep/len(train_dl):.4f}, train_acc: {train_correct/train_samples:.4f} , val_loss: {loss_val/len(val_dl):.4f} , val_acc: {val_correct/val_samples:.4f}')
    
        history["train_loss"].append(loss_ep/len(train_dl))
        history["train_acc"].append(train_correct.item()/train_samples)
        history["val_loss"].append(loss_val/len(val_dl))
        history["val_acc"].append(val_correct.item()/val_samples)
    
    # Print accuracy
    print(f'Accuracy for fold {k}: {float(val_correct) / float(val_samples) * 100:.2f} %')
    print(f'Loss for fold {k}: {float(loss_val)/float(len(val_dl)):.4f}')
    
    torch.save(model.state_dict(), "models/" + architecture + '/x' + str(patch_size) + '/' + str(N) + '/' + mode + '/' + architecture + '_x'+ str(patch_size) + "_k" + str(k) + ".pt") #SAVES THE TRAINED MODEL
    print('Model saved')   
    print('--------------------------------')
    results[k] = 100.0 * (val_correct / val_samples)
    losses[k] = loss_val/len(val_dl)

    # TESTING     
    
    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, num_epochs+1)

    plot_losses(epochs,history, architecture, patch_size, k,N)
    plot_accuracy(epochs,history, architecture, patch_size, k, N)
    
    history["train_loss"].clear()
    history["train_acc"].clear()
    history["val_loss"].clear()
    history["val_acc"].clear()

    # Validation...
    which_class = 1
    device='cpu'
    model = model.to(device)
    actuals, class_probabilities = test_class_probabilities(model, device, val_dl, which_class)
    fpr, tpr, _ = roc_curve(actuals, class_probabilities)
    roc_auc = auc(fpr, tpr)
    roc_curve_data = (fpr, tpr, roc_auc)
    roc_curves.append(roc_curve_data)
    
    # Confusion Matrix
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

    for batch_idx, (data, sub, targets) in enumerate(val_dl):
        data = data.to(device)
        targets = targets.to(device)
        ## Forward Pass
        scores = model(data)
        _, predictions = scores.max(1)

        # Append batch prediction results
        predlist = torch.cat([predlist, predictions.view(-1).cpu()])
        lbllist = torch.cat([lbllist, targets.view(-1).cpu()])

    class_names = ['nonLesion', 'lesion']

    cm = confusion_matrix(lbllist.numpy(), predlist.numpy())
    confusion_matrices.append(cm)

    
    
# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
     print(f'Fold {key}: {value:.2f} %')
     sum += value
print(f'Average accuracy: {(sum/len(results.items())):.2f} %')
sum = 0
for key, value in losses.items():
     print(f'Fold {key}: {value:.4f}')
     sum += value
print(f'Average loss: {(sum/len(losses.items())):.4f}')

