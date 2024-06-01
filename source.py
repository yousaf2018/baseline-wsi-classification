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

# Visualization function
def visualize_samples(loader, num_samples=5):
    data_iter = iter(loader)
    images, labels = data_iter.next()
    images = images[:num_samples]
    labels = labels[:num_samples]

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for idx, ax in enumerate(axes):
        image = images[idx].numpy().transpose((1, 2, 0))
        image = (image * 0.1) + 0.5  # Unnormalize
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"Label: {labels[idx].item()}")
    plt.show()

# Visualize some training samples
print("Training samples:")
visualize_samples(train_loader)

# Visualize some testing samples
print("Testing samples:")
visualize_samples(test_loader)

# open output file,
fconv = open(os.path.join(output, 'train_convergence.csv'), 'w')
fconv.write('epoch,loss,accuracy\n')
fconv.close()
fconv = open(os.path.join(output, 'valid_convergence.csv'), 'w')
fconv.write('epoch,tile-acc,max-auc,auc-avg-prob,auc-mjvt,auc-best\n')
fconv.close()

num_tiles = len(train_dset.slideIDX)

# making trainset of all tiles of training set slides
train_dset.maketraindata(np.arange(num_tiles))


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

        running_loss += loss.item()
        running_acc += calculate_accuracy(output, target)

    epoch_loss = running_loss / len(loader)
    epoch_acc = running_acc / len(loader)
    return epoch_loss, epoch_acc


def inference(run, loader, model):
    model.eval()
    correct = 0
    probs = []
    targets = []
    slideIDXs = []
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)

            prob = F.softmax(output, dim=1)[:, 1]
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            probs.extend(prob.data.cpu().numpy())
            targets.extend(target.data.cpu().numpy())
            slideIDXs.extend(target.data.cpu().numpy())

    probs = np.array(probs)
    targets = np.array(targets)
    slideIDXs = np.array(slideIDXs)

    acc = correct / len(loader.dataset)
    max_auc = calc_roc_auc(targets, group_max(slideIDXs, probs, loader.dataset.slideIDX))
    avg_prob_auc = calc_roc_auc(targets, group_avg(slideIDXs, probs))
    mj_vote_auc = calc_roc_auc(targets, group_max(slideIDXs, pred, loader.dataset.slideIDX))

    print("\nTest set: AUCs: Max: {:.4f}, Avg_Prob: {:.4f}, Maj_Vote: {:.4f}, \n".format(max_auc, avg_prob_auc,
                                                                                         mj_vote_auc))
    return acc, max_auc, avg_prob_auc, mj_vote_auc


for epoch in range(nepochs):
    # train for one epoch
    train_loss, train_acc = train(epoch, train_loader, model, criterion, optimizer)
    print('Epoch: [{}/{}], Training: Loss: {:.4f}, Accuracy: {:.2f}%\n'.format(epoch + 1, nepochs, train_loss,
                                                                              100 * train_acc))

    # write epoch's results
    fconv = open(os.path.join(output, 'train_convergence.csv'), 'a')
    fconv.write('{},{:.4f},{:.4f}\n'.format(epoch + 1, train_loss, train_acc))
    fconv.close()

    if epoch % test_every == 0:
        acc, max_auc, avg_prob_auc, mj_vote_auc = inference(epoch, test_loader, model)
        fconv = open(os.path.join(output, 'valid_convergence.csv'), 'a')
        fconv.write('{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(epoch + 1, acc, max_auc, avg_prob_auc, mj_vote_auc,
                                                                     max(mj_vote_auc, avg_prob_auc)))
        fconv.close()
        if max_auc > best_auc_v:
            best_auc_v = max_auc
            torch.save(model.state_dict(), os.path.join(output, 'best_epoch.pth'))

        torch.save(model.state_dict(), os.path.join(output, 'latest_epoch.pth'))

print('Training complete.')