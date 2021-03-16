import tez
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid

from sklearn.metrics import accuracy_score, confusion_matrix
import os
from tqdm import tqdm

from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision


class ImageClassifier(tez.Model):
    def __init__(self, num_classes):
        super().__init__()

        self.effnet = EfficientNet.from_pretrained("efficientnet-b0")
        
        self.dropout = nn.Dropout(0.2)
        self.in_features = self.effnet._fc.in_features
        self.out = nn.Linear(self.in_features, num_classes)
        self.step_scheduler_after = "epoch"

        # self.model = torchvision.models.resnet18(pretrained=True)
        # self.model = nn.Sequential(*list(self.model.children())[:-1])
        # self.in_features = 512
        # for param in self.model.parameters():
        #     param.requires_grad = False

        for param in self.effnet.parameters():
            param.requires_grad = False

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=3e-4)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return sch

    def forward(self, image, targets=None):
        batch_size, _, _, _ = image.shape

        x = self.effnet.extract_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        outputs = self.out(self.dropout(x))

        # x = self.model(image)
        # x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        # outputs = self.out(self.dropout(x))

        if targets is not None:
            loss = nn.CrossEntropyLoss()(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, None, None

class ImageDataset:
    def __init__(self, images, targets=None):
        self.images = images
        self.targets = targets
        self.aug = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # images = torch.tensor(self.images[idx], dtype=torch.float)
        targets = torch.tensor(self.targets[idx], dtype=torch.long)
        images = self.aug(Image.fromarray(self.images[idx]))
        if self.targets is not None:
            data = {
                "image": images,
                "targets": targets
            }
        else:
            data = {"image": images}
        return data


def plot_confusion_matrix(model, ds, labels):
    '''
    This function plots the confusion matrices given y_true, y_pred, labels.
    '''
    model.eval();
    model.cuda();
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for idx, data in enumerate(loader):
            for k, v in data.items():
                data[k] = v.to('cuda')
            pred, _, _ = model(**data)
            y_true.extend(list(data['targets'].detach().cpu().numpy()))
            y_pred.extend(list(torch.argmax(pred, dim=1).detach().cpu().numpy()))

    C = confusion_matrix(y_true, y_pred)
    A =(((C.T)/(C.sum(axis=1))).T)
    B =(C/C.sum(axis=0))
    plt.figure(figsize=(12,9))
    ax = sns.heatmap(A*100, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, annot_kws={"size": 18}, cmap="YlGnBu")
    for i, t in enumerate(ax.texts):
        if i % (len(labels) + 1) == 0:
            t.set_text(str(C.reshape(-1)[i]) + "/" + str(C.sum(axis=1)[i//len(labels)])) 
        else:
            t.set_text(str(C.reshape(-1)[i]))    
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title("Confusion matrix")
    plt.show()
    # plt.savefig(path)
    # plt.savefig(path, bbox_inches='tight')

def show_images(images, nmax=16):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for data in dl:
        show_images(data['image'], nmax)
        break

def prediction(model, ds, stage):
    model.eval();
    model.cuda();
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for idx, data in enumerate(loader):
            for k, v in data.items():
                data[k] = v.to('cuda')
            pred, _, _ = model(**data)
            y_true.extend(list(data['targets'].detach().cpu().numpy()))
            y_pred.extend(list(torch.argmax(pred, dim=1).detach().cpu().numpy()))
    print(f"Accuracy on {stage}: {accuracy_score(y_true, y_pred)*100:.2f}%")
    # return accuracy_score(y_true, y_pred), y_true, y_pred


def most_confused_classes(model, ds, labels):

    class2idx = dict([[i, c] for i, c in enumerate(labels)])
    dx2class = dict([[c, i] for i, c in enumerate(labels)])

    model.eval();
    model.cuda();
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for idx, data in enumerate(loader):
            for k, v in data.items():
                data[k] = v.to('cuda')
            pred, _, _ = model(**data)
            y_true.extend(list(data['targets'].detach().cpu().numpy()))
            y_pred.extend(list(torch.argmax(pred, dim=1).detach().cpu().numpy()))

    clsr = classification_report(y_true, y_pred, output_dict=True)
    clsr = sorted([[int(k), round(v['recall'], 2)] for k, v in clsr.items() if k.isnumeric()], key = lambda x: x[1])
    mcc = [i[0] for i in clsr][:2]

    out = []
    count = 0
    for data in ds:
        row = {}
        if int(data['targets']) in mcc:
            data['image'] = data['image'].unsqueeze(0).cuda()
            data['targets'] = data['targets'].unsqueeze(0).cuda()

            pred, _, _ = model(**data)
            pred = torch.argmax(pred, dim=1).detach().cpu().numpy()[0]
            y = int(data['targets'].detach().cpu().numpy()[0])
            if pred != y:
                row["image"] = data['image'].permute(0, 2, 3, 1).detach().cpu().numpy()[0]
                row["target"] = y
                row["predicted"] = int(pred)
                count += 1
                out.append(row)
        if count == 8:
            break

    fig = plt.figure(figsize=(20, 12))

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                    axes_pad=0.5,  # pad between axes in inch.
                    )

    for ax, (i, im) in zip(grid, enumerate(out)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im['image'])
        ax.set_title(f"Acutual Class: {class2idx[im['target']]} / Predicted Class: {class2idx[im['predicted']]}")


    plt.show()

