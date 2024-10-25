import torchvision.transforms as transforms
from PIL import Image
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch.nn.functional as F

class Sharpen(object):
    def __call__(self, img):
        img = np.array(img)
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        return Image.fromarray(sharpened)

transform = transforms.Compose(
    [
     transforms.Resize((227,227)),
     #transforms.RandomHorizontalFlip(p=0.7),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
     ]
    )  

train_transform = transforms.Compose(
    [
     transforms.Resize((227,227)),
     #transforms.RandomCrop(227),
     transforms.RandomHorizontalFlip(), 
     transforms.RandomVerticalFlip(),
     transforms.RandomRotation(90),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
     ]
    )  

transform_sharpen = transforms.Compose(
    [
     transforms.Resize((227,227)),
     #transforms.RandomHorizontalFlip(p=0.7),
     Sharpen(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
     ]
    )  

train_transform_sharpen = transforms.Compose(
    [
     transforms.Resize((227,227)),
     #transforms.RandomCrop(227),
     Sharpen(),
     transforms.RandomHorizontalFlip(), 
     transforms.RandomVerticalFlip(),
     transforms.RandomRotation(90),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
     ]
    )  


class EpilepsyDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(os.path.join(root_dir, annotation_file))
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        direc=os.path.join(self.root_dir, img_id).replace('\\','/')
        img = Image.open(direc).convert("RGB")
        sub = torch.tensor(int(self.annotations.iloc[index,1]))
        y_label = torch.tensor(float(self.annotations.iloc[index, 2]))
        
        if self.transform is not None:
            img = self.transform(img)

        return (img, sub, y_label)

    def get_subset(self, subject_values):
        samples = [self[i] for i in range(len(self)) if self.annotations.iloc[i, 1] in subject_values]
        return samples


def test_class_binary(model, device, test_loader, which_class, threshold=0.5):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, sub, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = F.softmax(output, dim=1)[:, which_class]
            predicted_class = (probabilities > threshold).float()  # Consideramos positivas aquellas con probabilidad mayor que el umbral
            actuals.extend(target == which_class)
            predictions.extend(predicted_class)
    return [i.item() for i in actuals], [i.item() for i in predictions]

def test_class_probabilities(model, device, test_loader, which_class):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data, sub, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities_batch = F.softmax(output, dim=1)  # Compute probabilities
            probabilities.extend(probabilities_batch[:, which_class])  # Extract probabilities for the specified class
            actuals.extend(target == which_class)  # Compare target with the specified class
    return [i.item() for i in actuals], [i.item() for i in probabilities]

def EpilepsyDet(label):
    if (label == 0):
        res = "nonLesion"
    else:
        res = "lesion"
    return res