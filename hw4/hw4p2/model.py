import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.autograd import Variable
from dataloader import train_loader, dev_loader

device = torch.device('cuda')

class PBLSTM(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(PBLSTM, self).__init__()
		self.lstm = nn.LSTM(input_size*2, hidden_size, 1, batch_first=True, bidirectional=True)

	def forward(self, x, x_lengths): # x.shape(B,T,feat_dim)
		if x.shape[1] % 2 != 0:
			x = x[:, :-1, :]

		x = x.contiguous().view(x.shape[0], int(x.shape[1]/2), x.shape[2]*2)
		x_pack = pack_padded_sequence(x, x_lengths, batch_first=True)
		output, _ = self.lstm(x_pack)
		output = pad_packed_sequence(output, batch_first=True)
		return output[0]

class Listener(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(Listener, self).__init__()
		self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=True)
		self.pblstm_1 = PBLSTM(hidden_size*2, hidden_size)
		self.pblstm_2 = PBLSTM(hidden_size*2, hidden_size)
		self.pblstm_3 = PBLSTM(hidden_size*2, hidden_size)
		self.key = nn.Linear(512, 128)  # From Listener
		self.value = nn.Linear(512, 128) # From Listener

	def forward(self, x, x_lengths):
		x_pack = pack_padded_sequence(x, x_lengths, batch_first=True)
		output, _ = self.lstm(x_pack)
		output = pad_packed_sequence(output, batch_first=True)
		output = output[0]
		seq_len = [int(x/2) for x in x_lengths]
		output = self.pblstm_1(output, seq_len)
		seq_len = [int(x/4) for x in x_lengths]
		output = self.pblstm_2(output, seq_len)
		seq_len = [int(x/8) for x in x_lengths]
		output = self.pblstm_3(output, seq_len)

		key = self.key(output)
		value = self.value(output)	
		return key, value, seq_len

class Attention(nn.Module):
	def __init__(self):
		super(Attention, self).__init__()
		self.softmax = nn.Softmax(dim=2)
		self.query = nn.Linear(512, 128) # From the last hidden unit in Speller

	def forward(self, key, value, speller_hidden, seq_len):
		query = self.query(speller_hidden)
		energy = torch.bmm(query.unsqueeze(1), key.permute(0, 2, 1))
		mask = Variable(torch.zeros(energy.shape[0],energy.shape[1],energy.shape[2]), requires_grad=False).to(device)

		for i in range(energy.shape[0]):
			length = seq_len[i]
			if length >= energy.shape[2]:
				mask[i, 0, :] = 1
			else:
				mask[i, 0, :length] = 1

		masked_attention = mask * self.softmax(energy)

		# Normalize: because masked_attention = Normalize[softmax(energy) * mask]
		masked_attention = F.normalize(masked_attention,p=1, dim=2)
		context = torch.bmm(masked_attention, value) # shape(batch_size, context_size)
		return context.squeeze(1)

class Speller(nn.Module):
	def __init__(self, attention):
		super(Speller, self).__init__()
		self.lstm_cell1 = nn.LSTMCell(640, 512)
		self.lstm_cell2 = nn.LSTMCell(512, 512)
		self.activation = nn.ELU()
		self.embedding = nn.Embedding(34, 512)
		self.fc1 = nn.Linear(640, 512)
		self.fc2 = nn.Linear(512, 34)
		self.attention = attention
		self.fc2.weight = self.embedding.weight

	def single_step(self, key, value, seq_len, hidden, cell, char_out):
		emb = self.embedding(char_out) # shape(batch_size, speller_hidden_size)
		context = self.attention(key, value, hidden[-1], seq_len)
		inp = torch.cat([emb, context], 1)

		hidden_out = []
		cell_out = []

		l1_hidden_out, l1_cell_out = self.lstm_cell1(inp, (hidden[0], cell[0]))
		hidden_out.append(l1_hidden_out)
		cell_out.append(l1_cell_out)

		l2_hidden_out, l2_cell_out = self.lstm_cell2(l1_hidden_out, (hidden[1], cell[1]))
		hidden_out.append(l2_hidden_out)
		cell_out.append(l2_cell_out)

		out = torch.cat([l2_hidden_out, context], 1)
		out = self.fc1(out)
		out = self.activation(out)
		out = self.fc2(out)
		return out, hidden_out, cell_out

	def forward(self, batch_size, key, value, seq_len, teacher_force, labels):
		#Get initial hidden & cell for LSTMCell
		hidden = [torch.randn(1, 512).to(device), torch.randn(1, 512).to(device)]
		hidden = [x.repeat(batch_size,1) for x in hidden] # 64 is batch size
		cell = [torch.randn(1, 512).to(device), torch.randn(1, 512).to(device)]
		cell = [x.repeat(batch_size,1) for x in cell]
		char = torch.ones(batch_size).long().to(device) # initial outputs are all 'SOS'

		iteration = labels.shape[1] if labels is not None else 250 # max iteration

		seq_pred_out = [] # containing char predictions (sequences) for each utterance in one batch, after stacking, final shape should be (batch_size, iteration, num_classes): (28, iteration, 34)
		for i in range(iteration):
			out, hidden, cell = self.single_step(key, value, seq_len, hidden, cell, char)
			seq_pred_out.append(out)

			# Teacher Forcing
			random = np.random.random()
			if labels is None or random < teacher_force:
				_, char = torch.max(out, 1)
			else:
				char = labels[:, i].long()
		return torch.stack(seq_pred_out, 1)

class LAS(nn.Module):
	def __init__(self):
		super(LAS, self).__init__()
		self.listener = Listener(40, 256)
		self.attention = Attention()
		self.speller = Speller(self.attention)

	def forward(self, batch_size, x, x_lengths, teacher_force=0.1, labels=None):
		key, value, seq_len = self.listener(x, x_lengths)
		output = self.speller(batch_size=batch_size, key=key, value=value, seq_len=seq_len, teacher_force=teacher_force, labels=labels)
		return output

model = LAS()
model = model.to(device)