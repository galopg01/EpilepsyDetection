from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from dataset import *
from utilities import *
import json
import argparse


parser = argparse.ArgumentParser(description="Training script")

parser.add_argument('--architecture', type=str, default="alexnet", help='Model architecture')
parser.add_argument('--k', type=int, default=5, help='Folds for cross validation')
parser.add_argument('--N', type=int, default=1, help='Number of positive pixels')
parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
parser.add_argument('--mode', type=str, default="normal", help='Mode')

args = parser.parse_args()

architecture = args.architecture
patch_size= args.patch_size
mode = args.mode
N= args.N
k_folds = args.k

mod = mode
if mode == "normal":
 mod=''

if mod=='':
    json_fileN = "./metrics/resultsN-" + str(patch_size) + ".json"
else:
    json_fileN = "./metrics/resultsN-" + str(patch_size) + "-" + mod + ".json"

test_data = EpilepsyDataset("","./testData/" + str(N) + "/" + mode + "/testx" + str(patch_size) + ".csv",transform=transform_sharpen)
test_ds = test_data

test_dl = DataLoader(test_ds, shuffle=False, batch_size=64)

confusion_matrices = []
roc_curves = []


if os.path.exists(json_fileN):
    with open(json_fileN, 'r') as file:
        results_jsonN = json.load(file)
else:
    results_jsonN = {}

auc_list = []
fscore_list = []

precision_list = []
recall_list = []
accuracy_list = []

for k in range(k_folds):
    
    print(f'FOLD {k}')
    print('--------------------------------')
    
    print("Loading model...")
    if (architecture=="alexnet"):
        model = torchvision.models.alexnet()
    elif (architecture == "mobilenet"):
        model = torchvision.models.mobilenet_v3_small()
    else:
        model = torchvision.models.shufflenet_v2_x1_5()
        
    model.load_state_dict(torch.load("models/" + architecture + "/x" + str(patch_size) + "/" + str(N) + "/" + mode +  "/" + architecture + "_x" + str(patch_size) + "_k" + str(k) + ".pt")) #loads the trained model
    model.eval()
    
    num_correct = 0
    num_samples = 0

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    which_class = 1
    device='cpu'
    actuals, class_probabilities = test_class_probabilities(model, device, test_dl, which_class)
    fpr, tpr, _ = roc_curve(actuals, class_probabilities)
    roc_auc = auc(fpr, tpr)

    roc_curve_data = (fpr, tpr, roc_auc)
    roc_curves.append(roc_curve_data)
    
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
    
    for batch_idx, (data, sub, targets) in enumerate(test_dl):
        data = data.to(device)
        targets = targets.to(device)
        ## Forward Pass
        scores = model(data)
        _, predictions = scores.max(1)

        # Append batch prediction results
        predlist=torch.cat([predlist,predictions.view(-1).cpu()])
        lbllist=torch.cat([lbllist,targets.view(-1).cpu()])
        
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)
        samples = predictions.size(0)

        for i in range(samples):
            targ = targets[i].item()
            pred = predictions[i].item()
            targ = EpilepsyDet(targ)
            pred = EpilepsyDet(pred)
            if (pred == targ):
                if (targ == "nonLesion"):
                    TN += 1
                else:
                    TP +=1
            elif (pred != targ):
                if (targ == "nonLesion"):
                    FP +=1
                else:
                    FN +=1
                    
    class_names = ['nonLesion', 'lesion']

    cm = confusion_matrix(lbllist.numpy(), predlist.numpy())
    confusion_matrices.append(cm)
    
    precision, recall, f_score, support = precision_recall_fscore_support(lbllist.numpy(), predlist.numpy(), average=None)
    balanced_accuracy = balanced_accuracy_score(lbllist.numpy(), predlist.numpy(), adjusted=False)
    print(f"True precision: ", precision.mean())
    print(f"True sensitivity (recall): ", recall.mean())
    #print("specificity: ", specificity)
    print("True F-score = ", f_score.mean())
    
  
    print("TP = ", TP, "TN = ", TN, "FP = ", FP, "FN = ", FN)
    accur = TP/(FP+TP)
    print(f"Precision: ", TP/(FP+TP))
    sensitivity = TP/(FN+TP)
    print(f"Sensitivity (recall): ", sensitivity)
    specificity = TN/(TN+FP)
    print("specificity: ", specificity)
    print("Balanced accuracy: ", (sensitivity + specificity)/2)
    fscore = TP/(TP+0.5*(FP+FN))
    print("F-score = ", fscore)
    print(
        f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    )

    auc_list.append(roc_auc)
    fscore_list.append(fscore)

    precision_list.append(accur)
    recall_list.append(sensitivity)
    accuracy_list.append(round(float(num_correct) / float(num_samples) * 100,2))

auc_mean = np.mean(auc_list)
fscore_mean = np.mean(fscore_list)

precision_mean = round(np.mean(precision_list), 2)
recall_mean = round(np.mean(recall_list), 2)
accuracy_mean = round(np.mean(accuracy_list), 2)
        
if architecture not in results_jsonN:
        results_jsonN[architecture] = {}
    

if N in results_jsonN[architecture]:
    results_jsonN[architecture][N]['AUC'] = auc_mean
    results_jsonN[architecture][N]['F-score'] = fscore_mean
else:
    results_jsonN[architecture][N] = {
        'AUC': auc_mean,
        'F-score': fscore_mean
    }

with open(json_fileN, 'w') as file:
    json.dump(results_jsonN, file, indent=4)

print("TESTING RESULTS")
print("-----------------------")
print("Precision mean:", precision_mean)
print("Sensitivity (Recall) mean:", recall_mean)
print("F-score mean:", fscore_mean)
print("Accuracy mean:", accuracy_mean)


show_roc_curves(roc_curves, architecture, patch_size, k, N)
show_confusion_matrices(confusion_matrices, architecture, patch_size, k, N)
