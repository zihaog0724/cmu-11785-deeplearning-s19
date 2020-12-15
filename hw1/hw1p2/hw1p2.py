import os
import copy
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle,check_random_state

device = torch.device('cuda')
os.environ['WSJ_PATH'] = '/home/ubuntu'
class WSJ():
    """ Load the WSJ speech dataset
        
        Ensure WSJ_PATH is path to directory containing 
        all data files (.npy) provided on Kaggle.
        
        Example usage:
            loader = WSJ()
            trainX, trainY = loader.train
            assert(trainX.shape[0] == 24590)
            
    """
  
    def __init__(self):
        self.dev_set = None
        self.train_set = None
        self.test_set = None
  
    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = load_raw(os.environ['WSJ_PATH'], 'dev')
        return self.dev_set

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_raw(os.environ['WSJ_PATH'], 'train')
        return self.train_set
  
    @property
    def test(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(os.environ['WSJ_PATH'], 'test.npy'), encoding='bytes'), None)
        return self.test_set
    
def load_raw(path, name):
    return (
        np.load(os.path.join(path, '{}.npy'.format(name)), encoding='bytes'), 
        np.load(os.path.join(path, '{}_labels.npy'.format(name)), encoding='bytes')
    )

## # Raw data processing
def data_processing(x, k):


    half = int(len(x) / 2)
    num_frame_1 = 0
    for i in range(len(x)-half):
        num_frame_1 += x[i].shape[0]

    num_frame_2 = 0
    for i in range(len(x)-half,len(x)):
        num_frame_2 += x[i].shape[0]

    processed_data = []
    
    #f = 0
    n = 0
    #res = np.zeros((num_frame_1,x[i].shape[1] * (2*k + 1))) 
    for i in range(len(x)-half):
        padding = np.pad(x[i],((k,k),(0,0)),'constant')
        for j in range(x[i].shape[0]):
            #processed_data[f] = padding[j:j+2*k+1].flatten()
            #f += 1
            res = padding[j:j+2*k+1].reshape(x[0].shape[1] * (2 * k + 1),)
            n += 1
            if n % 10000 == 0:
                print('finished:' + str(n))
            processed_data.append(res)

    #res = np.zeros((num_frame_2,x[i].shape[1] * (2*k + 1)))
    for i in range(len(x)-half,len(x)):
        padding = np.pad(x[i],((k,k),(0,0)),'constant')
        for j in range(x[i].shape[0]):
                        #processed_data[f] = padding[j:j+2*k+1].flatten()
                        #f += 1
            res = padding[j:j+2*k+1].reshape(x[0].shape[1] * (2 * k + 1),)
            n += 1
            if n % 10000 == 0:
                print('finished:' + str(n))
            processed_data.append(res)
    processed_data = np.vstack(processed_data)
    return processed_data

loader = WSJ()

print('Training data processing')
k = 9
trainX, trainY = loader.train # len(trainX) = 24590, total 15449191 frames
trainX = data_processing(trainX, k)
#trainY = [trainY[i].reshape(trainY[i].shape[0],1) for i in range(len(trainY))]
trainY = np.concatenate(trainY) # (15449191,1)

print('Validation data processing')
devX, devY = loader.dev #len(devX) = 1103, total 669294 frames
devX = data_processing(devX, k)
#devY = [devY[i].reshape(devY[i].shape[0],1) for i in range(len(devY))]
devY = np.concatenate(devY) # (669294,1)

print('Test data processing')
testX, testY = loader.test # len(testX) = 1, total 169656 frames
testX = data_processing(testX, k)
testY = np.zeros((testX.shape[0], 138))

## # Build custom dataset
class Custom_Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return len(self.X)

batch_size = 64
last_batch_size = trainY.shape[0] % batch_size

train_dataset = Custom_Data(X = trainX, y = trainY)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

dev_dataset = Custom_Data(X = devX, y = devY)
dev_loader = DataLoader(dev_dataset, batch_size = 1, shuffle = False)

test_dataset = Custom_Data(X = testX, y = testY)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

## # Build neuralnet model
class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(760,1600),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1600),
                                    nn.Dropout(p=0.3))

        self.layer2 = nn.Sequential(nn.Linear(1600,1280), 
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1280),
                                    nn.Dropout(p=0.3))

        self.layer3 = nn.Sequential(nn.Linear(1280,1024), 
                                    nn.ReLU(),
                                    nn.BatchNorm1d(1024),
                                    nn.Dropout(p=0.2))

        self.layer4 = nn.Sequential(nn.Linear(1024,768),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(768),
                                    nn.Dropout(p=0.2))

        self.layer5 = nn.Sequential(nn.Linear(768,512),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(512),
                                    nn.Dropout(p=0.1))
        self.layer6 = nn.Sequential(nn.Linear(512,256),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(256),
                                    nn.Dropout(p=0.1))

        self.layer7 = nn.Linear(256,138)
        
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        return out

model = Net().to(device)

## # Loss function and Optimizer used in training
num_epochs = 18
learning_rate = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.3)

## # Train the model
print('Training')
training_loss = []
validation_loss = []
val_acc = []

best_acc = 0.0
best_wts = copy.deepcopy(model.state_dict())
for epoch in range(num_epochs):
    model.train()
    scheduler.step()
    total_train_loss = 0 # for all batches in a single epoch
    for i, (train_frame, train_label) in enumerate(train_loader):
        train_frame = train_frame.float().to(device)
        train_label = train_label.to(device)
        optimizer.zero_grad()
        out = model(train_frame)
        loss = criterion(out,train_label)
        loss.backward()
        optimizer.step()
        if (i+1) % 50000 == 0:
            print('epoch:'+str(epoch+1)+' batch:'+str(i+1))

        total_train_loss += loss.item()
    training_loss.append(total_train_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        correct = 0
        total_val_loss = 0
        for val_frame, val_label in dev_loader:
            val_frame = val_frame.float().to(device)
            val_label = val_label.to(device)
            val_out = model(val_frame)
            val_loss = criterion(val_out,val_label)
            total_val_loss += val_loss.item()
            _, prediction = torch.max(val_out, 1)       
            if val_label == prediction:
                correct += 1.0
        validation_loss.append(total_val_loss / len(dev_loader))
        val_acc.append(correct * 100 / len(dev_loader))
        if (correct * 100) / len(dev_loader) >= best_acc:
            best_acc = (correct * 100) / len(dev_loader)
            best_wts = copy.deepcopy(model.state_dict())
        print ('Epoch {}: Validation Accuracy is {}%'.format(epoch + 1, correct * 100 / len(dev_loader)))

print('Training loss:' + str(training_loss))
print('Validation loss:' + str(validation_loss))
print('Validation acc:' + str(val_acc))

csv_file = open('./submission.csv','w')
csv_write = csv.writer(csv_file,dialect='excel')
title = ['id','label']
csv_write.writerow(title)

print('Testing')
model.load_state_dict(best_wts)
model.eval()
with torch.no_grad():
    for i, (test_frame, test_label) in enumerate(test_loader):
        test_frame = test_frame.float().to(device)
        test_label = test_label.to(device)
        test_out = model(test_frame)
        _, pred = torch.max(test_out, 1)
        csv_write.writerow([str(i), str(int(pred))])
