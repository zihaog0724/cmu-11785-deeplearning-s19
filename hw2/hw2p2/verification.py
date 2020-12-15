import os
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def findNremove(path,pattern,maxdepth=1):
    """
    This function is to remove the weird files in our image datafolders.
    E.g. '.DS_Store' or files starting with '.'
    """
    cpath=path.count(os.sep)
    for r,d,f in os.walk(path):
        if r.count(os.sep) - cpath <maxdepth:
            for files in f:
                if files.startswith(pattern):
                    try:
                        #print "Removing %s" % (os.path.join(r,files))
                        os.remove(os.path.join(r,files))
                    except Exception as e:
                        print (e)
                    else:
                        print ("%s removed" % (os.path.join(r,files)))

findNremove('./',"._",5)
print('finished file removing')

device = torch.device('cuda')

print('Preparing Training Data')
train_data = torchvision.datasets.ImageFolder('./train_data/large/',transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(train_data,batch_size=256,shuffle=True,num_workers=8)

print('Preparing Validation Data')
val_data = torchvision.datasets.ImageFolder(root='./validation_classification/large/',transform=torchvision.transforms.ToTensor())
val_loader = DataLoader(val_data,batch_size=1,shuffle=False,num_workers=8)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel,out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel * 4))

        self.residual = None
        if in_channel != out_channel * 4 or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * 4))
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        out = self.block(x)
        if self.residual is None:
            out += x
        else:
            out += self.residual(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module): # ResNet50, 4 layers, with 3/4/6/3 residual blocks
    def __init__(self):
        super(ResNet, self).__init__()
        # First Conv layer before the residual parts
        self.pre_res = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True))

        # Residual parts in ResNet50
        self.res1 = self.layer_construction(in_channel=64, out_channel=64, stride=1, num_blocks=3)
        self.res2 = self.layer_construction(in_channel=256, out_channel=128, stride=2, num_blocks=4)
        self.res3 = self.layer_construction(in_channel=512, out_channel=256, stride=2, num_blocks=6)
        self.res4 = self.layer_construction(in_channel=1024,out_channel=512, stride=2, num_blocks=3)

        # Pooling layer and fc layer after the residual parts
        self.ave_pool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc1_classes = nn.Linear(2048,2300,bias=False)

        # learnable parameters initialization
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)

            if isinstance(m,nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def layer_construction(self, in_channel, out_channel, stride, num_blocks):
        """ Construct "layers" for ResNet50, which has 4 "layers".
            First layer has 3 residual blocks;
            Second layer has 4 residual blocks;
            Third layer has 6 residual blocks;
            Fourth layer has 3 residual blocks;
        """
        layer = [ResBlock(in_channel,out_channel,stride)]
        for i in range(0, num_blocks-1):
            layer.append(ResBlock(out_channel * 4, out_channel))

        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.pre_res(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.ave_pool(out)
        out = out.reshape(out.size(0),-1)
        classes_out = self.fc1_classes(out)
        return classes_out, out # out is face embedding, a vector of length 2048

model = ResNet().to(device)
model.load_state_dict(torch.load('./classification_best.pkl'))
for param in model.parameters():
    param.requires_grad = True
model.fc1_classes = nn.Linear(2048,4300)
model.to(device)

## # Define Hyperparameters
num_epochs = 9 
learning_rate = 1e-3
criterion_1 = nn.CrossEntropyLoss()
optimizer_1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=3, gamma=0.1)

## # Train the model
print('Training')
training_loss = []
validation_loss = []
val_acc = []

best_acc = 0.0
best_wts = copy.deepcopy(model.state_dict())
for epoch in range(num_epochs):
    model.to(device)
    model.train()
    scheduler.step()
    total_train_loss = 0 # for all batches in a single epoch
    for i, (train_img, train_label) in enumerate(train_loader):
        train_img = train_img.float().to(device)
        train_label = train_label.long().to(device)
        
        optimizer_1.zero_grad()
        
        train_classes, train_embeddings = model(train_img)

        loss = criterion_1(train_classes,train_label)
        loss.backward()
        
        optimizer_1.step()
    
        if (i+1) % 2000 == 0:
            print('epoch:'+str(epoch+1)+'|batch:'+str(i+1))

        total_train_loss += loss.item()

    print('epoch:'+str(epoch+1)+'|loss:'+str(total_train_loss / len(train_loader)))
    training_loss.append(total_train_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        correct = 0
        total_val_loss = 0
        for val_img, val_label in val_loader:
            val_img = val_img.float().to(device)
            val_label = val_label.long().to(device)
            
            val_classes, val_embeddings = model(val_img)
            
            val_loss = criterion_1(val_classes,val_label)
            total_val_loss += val_loss.item()
            
            _, prediction = torch.max(val_classes, 1)       
            if val_label == prediction:
                correct += 1.0

        validation_loss.append(total_val_loss / len(val_loader))
        val_acc.append(correct * 100 / len(val_loader))
        if (correct * 100) / len(val_loader) >= best_acc:
            best_acc = (correct * 100) / len(val_loader)
            best_wts = copy.deepcopy(model.state_dict())
        print ('Epoch {}: Validation Accuracy is {}%'.format(epoch + 1, correct * 100 / len(val_loader)))

torch.save(best_wts,'verification_best' + '.pkl')

print('Training loss:' + str(training_loss))
print('Validation loss:' + str(validation_loss))
print('Validation acc:' + str(val_acc))

## # Testing
print('Testing')
file = open('trials_test_new.txt','r')
img_name_sub = file.read().splitlines()
img_name = [i.split() for i in img_name_sub]

csv_file = open('./verification_submission.csv','w')
csv_write = csv.writer(csv_file,dialect='excel')

title = ['trial','score']
csv_write.writerow(title)

model.load_state_dict(best_wts)
model.eval()
with torch.no_grad():
    for i in range(len(img_name)):
        img_1 = Image.open('./test_veri_T_new/' + img_name[i][0])
        img_1 = torchvision.transforms.ToTensor()(img_1)
        img_1 = img_1.unsqueeze_(0)
        img_1 = img_1.float().to(device)
        img_2 = Image.open('./test_veri_T_new/' + img_name[i][1])
        img_2 = torchvision.transforms.ToTensor()(img_2)
        img_2 = img_2.unsqueeze_(0)
        img_2 = img_2.float().to(device)

        img1_classes, img1_embeddings = model(img_1)
        img2_classes, img2_embeddings = model(img_2)

        img1_embeddings = img1_embeddings.reshape(1,2048)
        img2_embeddings = img2_embeddings.reshape(1,2048)
        cos = nn.CosineSimilarity()
        output = cos(img1_embeddings, img2_embeddings)
        csv_write.writerow([img_name_sub[i], str(output.item())])
        if (i+1) % 10000 == 0:
            print('finished test ' + str(i+1))

"""
End of the verification task
"""
