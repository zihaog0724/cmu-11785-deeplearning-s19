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

## # Testing
print('Testing')
csv_file = open('./submission.csv','w')
csv_write = csv.writer(csv_file,dialect='excel')
title = ['Id','Predicted']
csv_write.writerow(title)

model.load_state_dict(torch.load('./val_best.pkl'))
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