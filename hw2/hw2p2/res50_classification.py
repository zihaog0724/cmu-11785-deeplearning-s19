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
train_data = torchvision.datasets.ImageFolder('./train_data/medium/',transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(train_data,batch_size=256,shuffle=True)

print('Preparing Validation Data')
val_data = torchvision.datasets.ImageFolder(root='./validation_classification/medium/',transform=torchvision.transforms.ToTensor())
val_loader = DataLoader(val_data,batch_size=1,shuffle=False)

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
#				nn.init.constant_(m.bias, 0.01)

			if isinstance(m,nn.Conv2d):
				nn.init.xavier_normal_(m.weight)
#				nn.init.constant_(m.bias,0.01)

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
		return classes_out

model = ResNet().to(device)

## # Define Hyperparameters
num_epochs = 12
learning_rate = 1e-3
criterion_1 = nn.CrossEntropyLoss()
optimizer_1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=6, gamma=0.1)

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
	for i, (train_img, train_label) in enumerate(train_loader):
		train_img = train_img.float().to(device)
		train_label = train_label.long().to(device)
		optimizer_1.zero_grad()	
		train_classes = model(train_img)
		loss = criterion_1(train_classes,train_label)
		loss.backward()
		
		optimizer_1.step()
		
		if (i+1) % 1000 == 0:
			print('epoch:'+str(epoch+1)+'|batch:'+str(i+1))

		total_train_loss += loss.item()

	print('epoch:'+str(epoch+1)+'|train loss:'+str(total_train_loss / len(train_loader)))
	training_loss.append(total_train_loss / len(train_loader))

	model.eval()
	with torch.no_grad():
		correct = 0
		total_val_loss = 0
		for val_img, val_label in val_loader:
			val_img = val_img.float().to(device)
			val_label = val_label.long().to(device)
			
			val_classes = model(val_img)
			
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

torch.save(best_wts,'classification_best' + '.pkl')

print('Training loss:' + str(training_loss))
print('Validation loss:' + str(validation_loss))
print('Validation acc:' + str(val_acc))

## # Testing
print('Testing')
file = open('test_order_classification.txt','r')
img_name = file.read().splitlines()
img_num = []
for i in img_name:
	img_num.append(i.strip('.jpg'))

## # Preparing Test Dataset
class ImageDataset(Dataset):
	def __init__(self, root_dir, file_list, target_list):
		self.root_dir = root_dir
		self.file_list = file_list
		self.target_list = target_list
		self.n_class = len(list(set(target_list)))

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):
		directory = os.path.join(self.root_dir, self.file_list[index])
		img = Image.open(directory)
		img = torchvision.transforms.ToTensor()(img)
		label = self.target_list[index]
		return img, label

test_data = ImageDataset('./test_classification/medium/', img_name, [0] * 4600)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

csv_file = open('./submission.csv','w')
csv_write = csv.writer(csv_file,dialect='excel')
title = ['id','label']
csv_write.writerow(title)

def find_keys(dict, idx): # dict is val/train_data.class_to_idx

	return list(dict.keys())[list(dict.values())[idx]]

model.load_state_dict(best_wts)
model.eval()
with torch.no_grad():
	for i, (test_img, test_label) in enumerate(test_loader):
		test_img = test_img.float().to(device)
		test_label = test_label.long().to(device)
		test_classes = model(test_img)
		_, pred = torch.max(test_classes, 1)
		true_pred = find_keys(val_data.class_to_idx, int(pred)) # String
		csv_write.writerow([img_num[i],true_pred])

"""
End of the classification task
"""
