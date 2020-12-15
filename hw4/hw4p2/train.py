import csv
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from dataloader import train_loader, dev_loader, test_loader, vocab
from model_without_gumbel_greedy import model

device = torch.device('cuda')

def conver_to_str(x):
    text = []
    for i in x:
        text.append(vocab[i.item()])
    return "".join(i for i in text)

def levenshtein(seq1, seq2):  
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def greedy_decode(model, key, value, seq_len):
    hidden = [torch.randn(1, 512).to(device), torch.randn(1, 512).to(device)]
    hidden = [x.repeat(1,1) for x in hidden]
    cell = [torch.randn(1, 512).to(device), torch.randn(1, 512).to(device)]
    cell = [x.repeat(1,1) for x in cell]
    char = torch.ones(1).long().to(device)

    chars = []
    for i in range(250):
        out, hidden, cell = model.speller.single_step(key, value, seq_len, hidden, cell, char)
        _, char = torch.max(out, 1)
        if char.item() == 0:
            break
        chars.append(char.item())

    if len(chars) == 0:
        chars = [0,0,0]
    chars = torch.LongTensor(chars).to(device)
    return chars

def plot_grad_flow(named_parameters, epoch, batch):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    ax = plt.gca()
    ax.tick_params(axis='x',which='major',labelsize=6)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.savefig(str(epoch+1) + "_" + str(batch+1))

batch_size = 28
num_epochs = 35
learning_rate = 1e-3
criterion = nn.CrossEntropyLoss(reduction='none')
criterion_dev = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=27, gamma=0.1)

print("Training")
training_loss = []
best_wts = copy.deepcopy(model.state_dict())
best_dis = 1e3
teacher_force = 0.1
for epoch in range(num_epochs):
    model.train()
    scheduler.step()
    total_train_loss = 0.0
    if epoch+1 > 5:
        teacher_force = 0.2
    if epoch+1 > 10:
        teacher_force = 0.3
    if epoch+1 > 15:
        teacher_force = 0.4
    if epoch+1 > 20:
        teacher_force = 0.5
    for batch, (seq, seq_lengths, labels, label_lengths) in enumerate(train_loader):
        seq = seq.to(device)
        seq_lengths = seq_lengths.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        optimizer.zero_grad()
        if batch == len(train_loader) - 1:
            num_data = seq.shape[0]
            output = model(batch_size=num_data, x=seq, x_lengths=seq_lengths, teacher_force=teacher_force, labels=labels)
        else:
            output = model(batch_size=batch_size, x=seq, x_lengths=seq_lengths, teacher_force=teacher_force, labels=labels)
    
        loss_matrix = criterion(output.permute(0,2,1),labels.long())
        mask = Variable(torch.zeros(loss_matrix.shape[0], loss_matrix.shape[1]), requires_grad=False).to(device)
        for i in range(mask.shape[0]):
            l = label_lengths[i]
            mask[i, :l] = 1
        masked_loss = loss_matrix * mask
        loss = torch.sum(masked_loss) / torch.sum(label_lengths)
        loss.backward()
        if (batch+1) % 200 == 0: 
            plot_grad_flow(model.named_parameters(), epoch, batch)
        optimizer.step()

        total_train_loss += loss.item()

        if (batch+1) % 200 == 0: 
            print("epoch: {} | batch: {} | train loss: {}".format(epoch+1, batch+1, loss.item()))

    print("EPOCH: {} | AVG TRAIN LOSS: {}".format(epoch+1, total_train_loss/len(train_loader)))
    training_loss.append(total_train_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        total_dev_loss = 0.0
        total_dev_perplexity = 0.0
        dis = 0.0
        for dev_batch, (dev_seq, dev_seq_lengths, dev_labels, dev_label_lengths) in enumerate(dev_loader):
            dev_seq = dev_seq.to(device)
            dev_seq_lengths = dev_seq_lengths.to(device)
            dev_labels = dev_labels.to(device)
            dev_label_lengths = dev_label_lengths.to(device)
            dev_key, dev_value, dev_seq_len = model.listener(dev_seq, dev_seq_lengths)
            dev_output = greedy_decode(model, dev_key, dev_value, dev_seq_len)
            dev_pred_str = conver_to_str(dev_output)
            dev_label_str = conver_to_str(dev_labels.squeeze(0).long())
            dis += levenshtein(dev_pred_str,dev_label_str)
            if (dev_batch+1) % 200 == 0:
                print(dev_pred_str)

        print("EPOCH: {} | DEV DIS: {}".format(epoch+1, dis/len(dev_loader)))
        if (dis/len(dev_loader)) <= best_dis:
            best_dis = dis/len(dev_loader)
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts,'val_best' + '.pkl')


print("Testing")
csv_file = open('./submission.csv','w')
csv_write = csv.writer(csv_file,dialect='excel')
title = ['Id','Predicted']
csv_write.writerow(title)

model.load_state_dict(best_wts)
model.eval()
with torch.no_grad():
    for test_batch, (test_seq, test_seq_lengths, test_labels, test_label_lengths) in enumerate(test_loader):
        test_seq = test_seq.to(device)
        test_seq_lengths = test_seq_lengths.to(device)
        test_labels = test_labels.to(device)
        test_label_lengths = test_label_lengths.to(device)
        test_key, test_value, test_seq_len = model.listener(test_seq, test_seq_lengths)
        test_chars = greedy_decode(model, test_key, test_value, test_seq_len)
        csv_write.writerow([str(test_batch),conver_to_str(test_chars)])
        print("finished: " + str(test_batch))
