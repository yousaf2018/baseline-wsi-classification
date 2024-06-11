import os
import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_lib = '/kaggle/working/baseline-wsi-classification/data/train.csv'
train_dir = '/kaggle/input/patched-dataset-wsi-for-kat/'

val_lib = ''
val_dir = ''

project = 'M1'             # Which MUTATION, choose from M1/M2/M3/M4 ...
output = os.path.join('/kaggle/working/baseline-wsi-classification/Output', project)
# Check if the output directory exists, if not, create it
if not os.path.exists(output):
    os.makedirs(output)
batch_size = 128
nepochs = 5

test_every = 1           # related to validation set, if needed
weights = 0.5            # weight of a positive class if imbalanced

lr = 1e-4                # learning rate
weight_decay = 1e-4      # l2 regularzation weight

best_auc_v = 0           # related to validation set, if needed


# data loader
class MPdataset(data.Dataset):
    def __init__(self, libraryfile='', path_dir=None, project=None, transform=None, mult=2):
        lib = pd.DataFrame(pd.read_csv(libraryfile, usecols=['SLIDES', project], keep_default_na=True))
        lib.dropna(inplace=True)

        tar = lib[project].values.tolist()
        allslides = lib['SLIDES'].values.tolist()
        slides = []
        tiles = []
        ntiles = []
        slideIDX = []
        targets = []
        j = 0
        for i, path in enumerate(allslides):
            t = []
            cpath = os.path.join(path_dir, str(path), "Small")
            for f in os.listdir(cpath):
                if '.jpg' in f:
                    t.append(os.path.join(cpath, f))
            if len(t) > 0:
                slides.append(path)
                tiles.extend(t)
                ntiles.append(len(t))
                slideIDX.extend([j] * len(t))
                targets.append(int(tar[i]))
                j += 1

        self.slides = slides
        self.slideIDX = slideIDX
        self.ntiles = ntiles
        self.tiles = tiles
        self.targets = targets
        self.transform = transform
        self.mult = mult
        self.mode = None
        self.t_data = []  # Initialize this to avoid issues in len

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x], self.tiles[x], self.targets[self.slideIDX[x]]) for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self, index):
        if self.mode == 1:
            tile = self.tiles[index]
            img = Image.open(str(tile)).convert('RGB')
            slideIDX = self.slideIDX[index]
            target = self.targets[slideIDX]
            if self.mult != 1:
                img = img.resize((224, 224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        elif self.mode == 2:
            slideIDX, tile, target = self.t_data[index]
            img = Image.open(str(tile)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224, 224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode == 1:
            return len(self.tiles)
        elif self.mode == 2:
            return len(self.t_data)


def calc_roc_auc(target, prediction):
    fpr, tpr, thresholds = roc_curve(target, prediction)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def calculate_accuracy(output, target):
    preds = output.max(1, keepdim=True)[1]
    correct = preds.eq(target.view_as(preds)).sum()
    acc = correct.float() / preds.shape[0]
    return acc


# function to calculate mean of data grouped per slide, used for aggregating tile scores into slide score
def group_avg(groups, data):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    unames, idx, counts = np.unique(groups, return_inverse=True, return_counts=True)
    group_sum = np.bincount(idx, weights=data)
    group_average = group_sum / counts
    return group_average


# function to find index of max value in data grouped per slide
def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out


# baseline cnn model to fine tune
model = models.resnet18(True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.cuda()

if weights == 0.5:
    criterion = nn.CrossEntropyLoss().cuda()
else:
    w = torch.Tensor([1 - weights, weights])
    criterion = nn.CrossEntropyLoss(w).cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lr)

cudnn.benchmark = True

# normalization
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.1, 0.1, 0.1])

trans = transforms.Compose([
    transforms.ToTensor(),
    normalize, ])

trans_Valid = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# loading data using custom dataloader class
train_dset = MPdataset(train_lib, train_dir, project, trans)
train_dset.setmode(1)  # Set mode to ensure __len__ works correctly

# Split the dataset into 70% training and 30% test
train_indices, test_indices = train_test_split(np.arange(len(train_dset)), test_size=0.3, random_state=42)

# Create data loaders for training and testing
train_dset.maketraindata(train_indices)
train_loader = torch.utils.data.DataLoader(
    train_dset,
    batch_size=batch_size, shuffle=True,
    num_workers=6, pin_memory=False
)

test_dset = MPdataset(train_lib, train_dir, project, trans)
test_dset.setmode(2)
test_dset.maketraindata(test_indices)
test_loader = torch.utils.data.DataLoader(
    test_dset,
    batch_size=batch_size, shuffle=False,
    num_workers=6, pin_memory=False
)

# open output file
fconv = open(os.path.join(output, 'train_convergence.csv'), 'w')
fconv.write('epoch,loss,accuracy\n')
fconv.close()
fconv = open(os.path.join(output, 'test_convergence.csv'), 'w')
fconv.write('epoch,loss,accuracy\n')
fconv.close()

# Store metrics for visualization
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    running_acc = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input.size(0)
        acc = calculate_accuracy(output, target)
        running_acc += acc.item() * input.size(0)
        if i % 100 == 0:
            print("Train Epoch: [{:3d}/{:3d}] Batch number: {:3d}, Training: Loss: {:.4f}, Accuracy: {:.2f}%".
                  format(run + 1, nepochs, i + 1, running_loss / ((i + 1) * input.size(0)), (100 * running_acc) / ((i + 1) * input.size(0))))

    return running_loss / len(loader.dataset), running_acc / len(loader.dataset)


def inference(run, loader, model, criterion):
    model.eval()
    running_loss = 0.
    running_acc = 0.
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            running_loss += loss.item() * input.size(0)
            acc = calculate_accuracy(output, target)
            running_acc += acc.item() * input.size(0)
    return running_loss / len(loader.dataset), running_acc / len(loader.dataset)


# training the model
for epoch in range(nepochs):
    train_loss, train_acc = train(epoch, train_loader, model, criterion, optimizer)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    fconv = open(os.path.join(output, 'train_convergence.csv'), 'a')
    fconv.write('{},{:.4f},{:.4f}\n'.format(epoch, train_loss, train_acc))
    fconv.close()

    if epoch % test_every == 0:
        test_loss, test_acc = inference(epoch, test_loader, model, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        fconv = open(os.path.join(output, 'test_convergence.csv'), 'a')
        fconv.write('{},{:.4f},{:.4f}\n'.format(epoch, test_loss, test_acc))
        fconv.close()

        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss, 100 * test_acc))

# Visualization of Training and Test metrics
def plot_and_save(metric_values, metric_name, output_dir):
    plt.figure()
    epochs = range(1, len(metric_values) + 1)
    plt.plot(epochs, metric_values, 'b', label=metric_name)
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{metric_name}.png'))
    plt.close()

# Plot and save the metrics
plot_and_save(train_losses, 'Training Loss', output)
plot_and_save(train_accuracies, 'Training Accuracy', output)
plot_and_save(test_losses, 'Test Loss', output)
plot_and_save(test_accuracies, 'Test Accuracy', output)

# Save the final model
torch.save(model.state_dict(), os.path.join(output, 'final_model.pth'))

print('Training complete.')
