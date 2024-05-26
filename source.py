import os
import numpy as np
import pandas as pd
import time
import random
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import auc, roc_curve
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


train_lib = '/kaggle/working/baseline-wsi-classification/data/train.csv'
train_dir = '/kaggle/input/patched-dataset-wsi-for-kat/'

val_lib = ''
val_dir = ''

project='M1'             # Which MUTATION, choose from M1/M2/M3/M4 ...
output = os.path.join('/kaggle/working/baseline-wsi-classification/Output', project)
# Check if the output directory exists, if not, create it
if not os.path.exists(output):
    os.makedirs(output)
batch_size = 128
nepochs = 20

test_every = 1           # related to validation set, if needed
weights = 0.5            # weight of a positive class if imbalanced

lr = 1e-4                # learning rate
weight_decay = 1e-4      # l2 regularzation weight
 
best_auc_v = 0           # related to validation set, if needed


# data loader
class MPdataset(data.Dataset):
    def __init__(self, libraryfile='', path_dir=None, project=None, transform=None, mult=2):         
            
        lib = pd.DataFrame(pd.read_csv(libraryfile, usecols = ['SLIDES', project], keep_default_na=True))
        lib.dropna(inplace=True)
        
        tar = lib[project].values.tolist()
        allslides = lib['SLIDES'].values.tolist()       
        print("All slides -->", allslides)
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
                slideIDX.extend([j]*len(t))
                targets.append(int(tar[i]))
                j+=1
                
        print('Number of Slides: {}'.format(len(slides)))
        print('Number of tiles: {}'.format(len(tiles)))
        self.slides = slides
        self.slideIDX = slideIDX
        self.ntiles = ntiles
        self.tiles = tiles
        self.targets = targets
        self.transform = transform
        self.mult = mult
        self.mode = None

    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.tiles[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:                                     # loads all tiles from each slide sequentially for train/validatoin set
            tile = self.tiles[index]
            img = Image.open(str(tile)).convert('RGB')
            slideIDX = self.slideIDX[index]
            target = self.targets[slideIDX]
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target                       
        elif self.mode == 2:                                  # used when a different trainset is prepared e.g. with given tile index                     
            slideIDX, tile, target = self.t_data[index]
            img = Image.open(str(tile)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
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
    acc = correct.float()/preds.shape[0]
    return acc

#function to calculate mean of data grouped per slide, used for aggregating tile scores into slide score
def group_avg(groups, data):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    unames, idx, counts = np.unique(groups, return_inverse=True, return_counts=True)
    group_sum = np.bincount(idx, weights=data)
    group_average = group_sum / counts
    return group_average

#function to find index of max value in data grouped per slide
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

#baseline cnn model to fine tune
model = models.resnet18(True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.cuda()

if weights==0.5:
    criterion = nn.CrossEntropyLoss().cuda()
else:
    w = torch.Tensor([1-weights,weights])
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

#loading data using cutom dataloader class
train_dset = MPdataset(train_lib, train_dir, project, trans)
train_loader = torch.utils.data.DataLoader(
    train_dset,
    batch_size=batch_size, shuffle=False,
    num_workers=6, pin_memory=False)

if val_lib:
    val_dset = MPdataset(val_lib, val_dir, project, trans_Valid)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=6, pin_memory=False)

#open output file, 
fconv = open(os.path.join(output,'train_convergence.csv'), 'w')
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

        running_loss += loss.item()*input.size(0)
        acc = calculate_accuracy(output, target)
        running_acc += acc.item()*input.size(0)
        if i%100 == 0:
            print("Train Epoch: [{:3d}/{:3d}] Batch number: {:3d}, Training: Loss: {:.4f}, Accuracy: {:.2f}%".
              format(run+1, nepochs, i+1, running_loss/((i+1)*input.size(0)), (100*running_acc)/((i+1)* input.size(0))))

    return running_loss/len(loader.dataset), running_acc/len(loader.dataset)

def inference(run, loader, model):
    model.eval()
    running_acc = 0.
    probs = torch.FloatTensor(len(loader.dataset))
    preds = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            acc = calculate_accuracy(output, target)
            y = F.softmax(output, dim=1)
            _, pr = torch.max(output.data, 1)
            preds[i * batch_size:i * batch_size + input.size(0)] = pr.detach().clone()
            probs[i * batch_size:i * batch_size + input.size(0)] = y.detach()[:, 1].clone()
            running_acc += acc.item() * input.size(0)
            if i % 100 == 0:
                print('Inference\tEpoch: [{:3d}/{:3d}]\tBatch: [{:3d}/{}]\t acc: {:0.2f}%'
                  .format(run + 1, nepochs, i + 1, len(loader), (100*running_acc)/((i+1)*input.size(0))))
    return probs.cpu().numpy(), running_acc/len(loader.dataset), preds.cpu().numpy()



# measure data loading time
start_time = time.time()

#loop throuh epochs
for epoch in range(nepochs):
    train_dset.shuffletraindata()
    train_dset.setmode(2)
    loss, acc = train(epoch, train_loader, model, criterion, optimizer)
    
    # measure elapsed time  so far  
    print("--- {:0.2f} minutes ---".format((time.time() - start_time)/60.))
    
    print('Training\tEpoch: [{}/{}]\tLoss: {:0.4f}\tAccuracy: {:0.4f}'.
          format(epoch+1, nepochs, loss, acc))
    
    fconv = open(os.path.join(output, 'train_convergence.csv'), 'a')
    fconv.write('{},{:0.4f},{:0.4f}\n'.format(epoch+1,loss, acc))
    fconv.close()

    #Validation if needed --- 
    if val_lib and (epoch+1) % test_every == 0:
        val_dset.setmode(1)
        val_probs, val_acc, val_preds = inference(epoch, val_loader, model)
        
        #aggregating tile scores into slide score - 3 different methods (max, average, and majority voting)
        aggregate_slide_predavg = group_avg(np.array(val_dset.slideIDX), val_preds)
        aggregate_slide_probavg = group_avg(np.array(val_dset.slideIDX), val_probs)
        aggregate_slide_max = group_max(np.array(val_dset.slideIDX), val_probs, len(val_dset.slides))
        
        fpr, tpr, thresholds = roc_curve(val_dset.targets, aggregate_slide_predavg)
        roc_auc_maj_vote = auc(fpr, tpr)
        fpr, tpr, thresholds = roc_curve(val_dset.targets, aggregate_slide_probavg)
        roc_auc_avg_prob = auc(fpr, tpr)
        fpr, tpr, thresholds = roc_curve(val_dset.targets, aggregate_slide_max)
        roc_auc_max_prob = auc(fpr, tpr)
        
        print('Validation\tEpoch: [{}/{}]\t val_acc: {:0.4f}\tROC-AUC: max_prob: {:0.4f}\t avg_prob: {:0.4f}\t maj_vote: {:0.4f}\t best so far: {:0.4f}'
              .format(epoch+1, nepochs, val_acc, roc_auc_max_prob, roc_auc_avg_prob, roc_auc_maj_vote, 
                      max(best_auc_v, roc_auc_max_prob, roc_auc_avg_prob, roc_auc_maj_vote)))
        
        fconv = open(os.path.join(output, 'valid_convergence.csv'), 'a')
        fconv.write('{},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f}\n'
                    .format(epoch+1, val_acc,roc_auc_max_prob, roc_auc_avg_prob, roc_auc_maj_vote, 
                            max(best_auc_v, roc_auc_max_prob, roc_auc_avg_prob, roc_auc_maj_vote)))
        
        #Save best model
        if max(roc_auc_max_prob, roc_auc_avg_prob, roc_auc_maj_vote) > best_auc_v:
            best_auc_v = max(roc_auc_max_prob, roc_auc_avg_prob, roc_auc_maj_vote)
            obj = {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_auc_v': best_auc_v,
                'optimizer' : optimizer.state_dict()
            }
            torch.save(obj, os.path.join(output,'checkpoint_best.pth'))
            
    # measure accumulated elapsed time so far 
    print("--- {:0.2f} minutes ---".format((time.time() - start_time)/60.))