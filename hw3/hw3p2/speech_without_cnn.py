import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import Levenshtein as L
import phoneme_list
import ctcdecode
import copy
import csv

device = torch.device('cuda')

trainX = np.load('wsj0_train.npy',encoding="latin1")
trainY = np.load('wsj0_train_merged_labels.npy',encoding="latin1")
trainY += 1.0
devX = np.load('wsj0_dev.npy',encoding="latin1")
devY = np.load('wsj0_dev_merged_labels.npy',encoding="latin1")
devY += 1.0
testX = np.load('transformed_test_data.npy',encoding="latin1")
testY = np.zeros((523,10))

def convert_to_string(tokens, vocab, seq_len):
	return ''.join([vocab[x] for x in tokens[0:seq_len]])

def collate(batch):
	descending = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
	sequence = [torch.Tensor(x[0]) for x in descending]
	label = [torch.Tensor(x[1]) for x in descending]
	seq_lengths = torch.LongTensor([len(x) for x in sequence])
	label_lengths = torch.LongTensor([len(x) for x in label])
	label = torch.cat(label)
	sequence_padded = pad_sequence(sequence)
	return sequence_padded, seq_lengths, label, label_lengths

class SpeechData(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y
	def __len__(self):
		return len(self.X)
	def __getitem__(self, index):
		return self.X[index], self.y[index]

train_dataset = SpeechData(X = trainX, y = trainY)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
dev_dataset = SpeechData(X = devX, y = devY)
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=collate)
test_dataset = SpeechData(X = testX, y = testY)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)

class SpeechModel(nn.Module):
	def __init__(self):
		super(SpeechModel,self).__init__()
		self.lstm = nn.LSTM(40,512,4,bidirectional=True)
		self.mlp = nn.Sequential(
			nn.Linear(1024,512),
			nn.ELU(),
			nn.Linear(512,47))

		for m in self.modules():
			if isinstance(m,nn.Linear):
				nn.init.xavier_normal_(m.weight)
				nn.init.constant_(m.bias, 0.01)	
			if isinstance(m,nn.LSTM):
				for param in m.parameters():
					if len(param.shape) >= 2:
						nn.init.orthogonal_(param.data)
					else:
						nn.init.normal_(param.data)

	def forward(self,x,x_lengths,label,label_lengths): #x.shape(T,B,40)
		x_pack = pack_padded_sequence(x, x_lengths)
		output, hidden = self.lstm(x_pack)
		output = pad_packed_sequence(output)
		output = self.mlp(output[0])
		return output, hidden

model = SpeechModel().to(device)

num_epochs = 18
learning_rate = 1e-3
criterion = nn.CTCLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.3)

## # Train the model
print('Training')
training_loss = []
val_loss = []
val_dis = []
best_dis = 1e20
best_wts = copy.deepcopy(model.state_dict())
for epoch in range(num_epochs):
	model.train()
	scheduler.step()
	total_train_loss = 0.0
	for i, (seq, seq_lengths, label, label_lengths) in enumerate(train_loader):
		seq = seq.to(device)
		seq_lengths = seq_lengths.to(device)
		label_lengths = label_lengths.to(device)
		label = label.long().to(device)
		optimizer.zero_grad()
		out, hidden = model(seq,seq_lengths,label,label_lengths)
		out = out.log_softmax(2)
		loss = criterion(out,label,seq_lengths,label_lengths)
		loss.backward()
		optimizer.step()
		
		if (i+1) % 100 == 0:
			print('epoch:'+str(epoch+1)+'|batch:'+str(i+1))
		total_train_loss += loss.item()
	
	print('epoch:' + str(epoch+1) + '|train loss:' + str(total_train_loss / len(train_loader)))
	training_loss.append(total_train_loss / len(train_loader))

	model.eval()
	with torch.no_grad():
		total_dev_loss = 0.0
		dis = 0.0
		for idx, (dev_seq, dev_seq_lengths, dev_label, dev_label_lengths) in enumerate(dev_loader):
			dev_seq = dev_seq.to(device)
			dev_seq_lengths = dev_seq_lengths.to(device)
			dev_label_lengths = dev_label_lengths.to(device)
			dev_label = dev_label.long().to(device)
			dev_out, dev_hidden = model(dev_seq,dev_seq_lengths,dev_label,dev_label_lengths) # dev_out.shape(T,B,out_channels)
			
			"""
			Using ctcdecode to decode the output, and using Levenshtein distance
			to compute the average distance(score).
			"""
			decode_out = dev_out.permute(1,0,2) # decode_out.shape(B,T,out_channels)
			decode_out = torch.cuda.FloatTensor(nn.Softmax(dim=2)(decode_out))
			decoder = ctcdecode.CTCBeamDecoder(phoneme_list.PHONEME_MAP, beam_width=2, blank_id=0)	
			beam_result, beam_scores, timesteps, out_seq_len = decoder.decode(decode_out)
			output_str = convert_to_string(beam_result[0][0], phoneme_list.PHONEME_MAP, out_seq_len[0][0])
			label_str = "".join([phoneme_list.PHONEME_MAP[i] for i in dev_label])
			dis += L.distance(output_str,label_str)

			"""
			Compute validation loss
			"""
			dev_out = dev_out.log_softmax(2)
			dev_loss = criterion(dev_out,dev_label,dev_seq_lengths,dev_label_lengths)
			total_dev_loss += dev_loss.item()
	
		avg_dis = dis / len(dev_loader)
		val_dis.append(avg_dis)

		if avg_dis <= best_dis:
			best_dis = avg_dis
			best_wts = copy.deepcopy(model.state_dict())
		print('epoch:' + str(epoch+1) + '|val dis:' + str(avg_dis))
		print('epoch:' + str(epoch+1) + '|val loss:' + str(total_dev_loss / len(dev_loader)))
		val_loss.append(total_dev_loss / len(dev_loader))

torch.save(best_wts,'val_best' + '.pkl')

print('Training loss:' + str(training_loss))
print('Validation loss:' + str(val_loss))
print('Validation Distance:' + str(val_dis))

## # Testing
print('Testing')
csv_file = open('./submission.csv','w')
csv_write = csv.writer(csv_file,dialect='excel')
title = ['Id','Predicted']
csv_write.writerow(title)

model.load_state_dict(best_wts)
model.eval()
with torch.no_grad():
	for i, (test_seq,test_seq_lengths,test_label,test_label_lengths) in enumerate(test_loader):
		test_seq = test_seq.to(device)
		test_seq_lengths = test_seq_lengths.to(device)
		test_label_lengths = test_label_lengths.to(device)
		test_label = test_label.long().to(device)
		test_out, test_hidden = model(test_seq,test_seq_lengths,test_label,test_label_lengths)	
			
		"""
		Using ctcdecode to decode the output, and using Levenshtein distance
		to compute the average distance(score).
		"""
		test_decode_out = test_out.permute(1,0,2) # decode_out.shape(B,T,out_channels)
		test_decode_out = torch.cuda.FloatTensor(nn.Softmax(dim=2)(test_decode_out))
		test_decoder = ctcdecode.CTCBeamDecoder(phoneme_list.PHONEME_MAP, beam_width=100, blank_id=0)	
		test_beam_result, test_beam_scores, test_timesteps, test_out_seq_len = test_decoder.decode(test_decode_out)
		test_output_str = convert_to_string(test_beam_result[0][0], phoneme_list.PHONEME_MAP, test_out_seq_len[0][0])
		csv_write.writerow([str(i),test_output_str])
