import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

def bytes2str(ndarr):
	Y = []
	for i in range(len(ndarr)):
		Y.append(ndarr[i].astype(str))
	Y = np.asarray(Y)
	return Y

def get_vocabulary(trainY, devY):
	vocab = [' ']
	for i in range(len(trainY)):
		for j in range(len(trainY[i])):
			for k in range(len(trainY[i][j])):
				if trainY[i][j][k] not in vocab:
					vocab.append(trainY[i][j][k])
	
	for i in range(len(devY)):
		for j in range(len(devY[i])):
			for k in range(len(devY[i][j])):
				if devY[i][j][k] not in vocab:
					vocab.append(devY[i][j][k])
	return vocab

def trans2ints(ndarr, vocab): # convert strings in transcripts to indices in vocab
	Y = []
	for i in range(len(ndarr)):
		transcripts = []
		for j in range(len(ndarr[i])):
			for k in range(len(ndarr[i][j])):
				idx = vocab.index(ndarr[i][j][k])
				transcripts.append(int(idx))
			transcripts.append(int(vocab.index(' ')))
		transcripts.append(0)
		Y.append(np.array(transcripts))
	return np.asarray(Y)

trainX = np.load('train.npy',encoding="latin1")
trainY = np.load('train_transcripts.npy',encoding="latin1")
devX = np.load('dev.npy',encoding="latin1")
devY = np.load('dev_transcripts.npy',encoding="latin1")
testX = np.load('test.npy',encoding="latin1")
testY = np.zeros((523,10))

trainY = bytes2str(trainY)
devY = bytes2str(devY)

vocab = get_vocabulary(trainY, devY)
vocab = sorted(vocab)
vocab.insert(0, '<eos>')
vocab.insert(1, '<sos>')

trainY = trans2ints(trainY, vocab)
devY = trans2ints(devY, vocab)

def collate(batch):
	descending = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
	sequence = [torch.Tensor(x[0]) for x in descending]
	label = [torch.Tensor(x[1]) for x in descending]
	seq_lengths = torch.LongTensor([len(x) for x in sequence])
	label_lengths = torch.LongTensor([len(x) for x in label])
	#label = torch.cat(label)
	sequence_padded = pad_sequence(sequence, batch_first=True)
	max_label_len = max(label_lengths)
	padded_labels = torch.zeros([len(label) ,max_label_len])
	for i in range(len(label)):
		label_len = label[i].shape[0]
		padded_labels[i, :label_len] = label[i]
	return sequence_padded, seq_lengths, padded_labels, label_lengths

class WSJData(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y
	def __len__(self):
		return len(self.X)
	def __getitem__(self, index):
		return self.X[index], self.y[index]

train_dataset = WSJData(X = trainX, y = trainY)
train_loader = DataLoader(train_dataset, batch_size=28, shuffle=True, collate_fn=collate)
dev_dataset = WSJData(X = devX, y = devY)
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=collate)
test_dataset = WSJData(X = testX, y = testY)
test_loader= DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)
