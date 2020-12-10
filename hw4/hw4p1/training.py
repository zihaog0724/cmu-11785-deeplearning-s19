
# coding: utf-8

# In[ ]:




import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation


# In[ ]:


# load all that we need

dataset = np.load('../dataset/wiki.train.npy')[0:2] # Training data shape(579,): 579 article, each article contains a vector of integers.
fixtures_pred = np.load('../fixtures/prediction.npz')  # dev
fixtures_gen = np.load('../fixtures/generation.npy')  # dev
fixtures_pred_test = np.load('../fixtures/prediction_test.npz')  # test
fixtures_gen_test = np.load('../fixtures/generation_test.npy')  # test
vocab = np.load('../dataset/vocab.npy')


# In[ ]:


# data loader

class LanguageModelDataLoader(DataLoader):
    """
        TODO: Define data loader logic here
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # concatenate your articles and build into batches
        
        if self.shuffle == True:
            np.random.shuffle(self.dataset)
        data = np.concatenate([i for i in self.dataset])
        nbatch = data.shape[0] // self.batch_size
        data = data[0: nbatch*self.batch_size]
        data = data.reshape(self.batch_size, nbatch).T
        
        i = 0
        while i < data.shape[0] - 1 - 1:
            mean = 140 if np.random.random() < 0.95 else 140 / 2.
            seq_len_init = max(5, int(np.random.normal(mean, 5)))
            seq_len = min(seq_len_init, data.shape[0] - 1 - i)
            x = data[i: i+seq_len]
            y = data[i+1: i+1+seq_len].ravel()
            yield torch.LongTensor(x), torch.LongTensor(y)
            i += seq_len_init     
            
#train_data = LanguageModelDataLoader(dataset, 64, shuffle=True)

#for epoch in range(2):
#    print("epoch:" + str(epoch))
#    for i, (x, y) in enumerate(train_data):
#        print(x.shape)
#        if i == 5:
#            break
        


# In[ ]:


#def embedding_dropout(embedding, data, dropout=0.1): # embedding = nn.Embedding()
#    if dropout == 0:
#        new_embedding_weight = embedding.weight
#    else:
#        new_embedding_weight = embedding.weight * (torch.randn(embedding.weight.shape[0], 1).bernoulli_(1 - dropout).expand_as(embedding.weight) / (1 - dropout))
#    return torch.nn.functional.embedding(data, new_embedding_weight)
    


# In[ ]:


#def locked_dropout(embedding, dropout=0.65):
#    if dropout == 0:
#        return embedding
#    else:
#        x = torch.randn(1, embedding.shape[1], embedding.shape[2]).bernoulli_(1 - dropout)
#        mask = Variable(m, requires_grad=False) / (1 - dropout)
#        return embedding * mask.expand_as(embedding)


# In[ ]:


# model

class LanguageModel(nn.Module):
    """
        TODO: Define your model here
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, nlayers):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.encoder = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, nlayers)
        self.decoder = nn.Linear(self.hidden_size, self.vocab_size)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -1, 1)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)	
            if isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)

    def forward(self, x):
        # Feel free to add extra arguments to forward (like an argument to pass in the hiddens)
        embedding = self.encoder(x)
        output, hidden = self.lstm(embedding)
        output = self.decoder(output)
        return output, hidden


# In[ ]:


# model trainer

class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id
        
        # TODO: Define your optimizer and criterion here
        self.optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train() # set to training mode
        epoch_loss = 0
        num_batches = 0
        for batch_num, (inputs, targets) in enumerate(self.loader):
            epoch_loss += self.train_batch(inputs, targets)
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs, self.max_epochs, epoch_loss))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        """ 
            TODO: Define code for training a single batch of inputs
        
        """
        self.optimizer.zero_grad()
        out, hidden = self.model(inputs)
        print(out.shape)
        print(out.view(-1,out.shape[2]).shape)
        print(targets.view(-1).shape)
        loss = self.criterion(out.view(-1,out.shape[2]), targets.view(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

    def test(self):
        # don't change these
        self.model.eval() # set to eval mode
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model) # get predictions
        self.predictions.append(predictions)
        generated_logits = TestLanguageModel.generation(fixtures_gen, 10, self.model) # generated predictions for 10 words
        generated_logits_test = TestLanguageModel.generation(fixtures_gen_test, 10, self.model)
        nll = test_prediction(predictions, fixtures_pred['out'])
        generated = test_generation(fixtures_gen, generated_logits, vocab)
        generated_test = test_generation(fixtures_gen_test, generated_logits_test, vocab)
        self.val_losses.append(nll)
        
        self.generated.append(generated)
        self.generated_test.append(generated_test)
        self.generated_logits.append(generated_logits)
        self.generated_logits_test.append(generated_logits_test)
        
        # generate predictions for test data
        predictions_test = TestLanguageModel.prediction(fixtures_pred_test['inp'], self.model) # get predictions
        self.predictions_test.append(predictions_test)
            
        print('[VAL]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs + 1, self.max_epochs, nll))
        return nll

    def save(self):
        # don't change these
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()},
            model_path)
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)), self.predictions[-1])
        np.save(os.path.join('experiments', self.run_id, 'predictions-test-{}.npy'.format(self.epochs)), self.predictions_test[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-{}.npy'.format(self.epochs)), self.generated_logits[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-test-{}.npy'.format(self.epochs)), self.generated_logits_test[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}-test.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated_test[-1])


# In[ ]:


class TestLanguageModel:
    def prediction(inp, model):
        """
            TODO: write prediction code here
            
            :param inp:
            :return: a np.ndarray of logits
        """
        raise NotImplemented

        
    def generation(inp, forward, model):
        """
            TODO: write generation code here

            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """        
        raise NotImplemented
        


# In[ ]:


# TODO: define other hyperparameters here

NUM_EPOCHS = 2
BATCH_SIZE = 64


# In[ ]:


run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)


# In[ ]:


model = LanguageModel(len(vocab),4, 10, 3)
loader = LanguageModelDataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
trainer = LanguageModelTrainer(model=model, loader=loader, max_epochs=NUM_EPOCHS, run_id=run_id)


# In[ ]:


best_nll = 1e30 
for epoch in range(NUM_EPOCHS):
    trainer.train()
    nll = trainer.test()
    if nll < best_nll:
        best_nll = nll
        print("Saving model, predictions and generated output for epoch "+str(epoch)+" with NLL: "+ str(best_nll))
        trainer.save()
    


# In[ ]:


# Don't change these
# plot training curves
plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.train_losses, label='Training losses')
plt.plot(range(1, trainer.epochs + 1), trainer.val_losses, label='Validation losses')
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.legend()
plt.show()


# In[ ]:


# see generated output
print (trainer.generated[-1]) # get last generated output

