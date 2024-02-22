#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install portalocker')
get_ipython().system('pip install torchmetrics')


# In[3]:


import argparse
import logging
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from torchtext.datasets import DATASETS
from torchtext.prototype.transforms import load_sp_model, PRETRAINED_SP_MODEL, SentencePieceTokenizer
from torchtext.utils import download_from_url
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torchtext.vocab import GloVe
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)


# ### Information
# - torchtext repo: https://github.com/pytorch/text/tree/main/torchtext
# - torchtext documentation: https://pytorch.org/text/stable/index.html

# In[3]:





# ### Constants

# In[4]:


DATASET = "AG_NEWS"
DATA_DIR = ".data"
DEVICE = "cuda"
EMBED_DIM = 300
LR = 4.0 ## will modify when training RCNN
BATCH_SIZE = 16
NUM_EPOCHS = 5
PADDING_VALUE = 0
PADDING_IDX = PADDING_VALUE


# In[4]:





# ### Get the tokenizer
# - Use the WordLevel tokenizer.
# 

# In[90]:


basic_english_tokenizer = get_tokenizer("basic_english")


# In[91]:


basic_english_tokenizer("This is some text ...")


# In[92]:


TOKENIZER = basic_english_tokenizer


# ### Get the data and get the vocabulary

# In[8]:


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield TOKENIZER(text)


# In[9]:


train_iter = DATASETS[DATASET](root=DATA_DIR, split="train")
VOCAB = build_vocab_from_iterator(yield_tokens(train_iter), specials=('<pad>', '<unk>'))

# Make the default index the same as that of the unk_token
VOCAB.set_default_index(VOCAB['<unk>'])


# ### Get GloVe embeddings ... This will be slow ...

# In[11]:


GLOVE = GloVe()


# In[12]:


len(GLOVE), GLOVE.vectors.shape


# ### Helper functions

# In[13]:


def text_pipeline(text):
    return VOCAB(TOKENIZER(text))

def label_pipeline(label):
    return int(label) - 1


# Nice link on collate_fn and DataLoader in PyTorch: https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3

# In[14]:


## -- left shift labels since indices start from 0, convert text to int64, and convert the processed labels and text(int)s to torch tensor, then move them to DEVICE

## -- the label and texts are processed separately, thus we don't need to zip them as (label, text) tuples.
## -- We may just loop through all the labels to do the left shifting, and loop through all the text to apply text_pipeline()
def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        # Get the label from {1, 2, 3, 4} to {0, 1, 2, 3}
        label_list.append(label_pipeline(_label))

        # Return a list of ints.
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text.clone().detach())

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True)

    return label_list.to(DEVICE), text_list.to(DEVICE)


# ### Get the data

# In[15]:


train_iter = DATASETS[DATASET](root=DATA_DIR, split="train")
num_class = len(set([label for (label, _) in train_iter]))
print(f"The number of classes is {num_class} ...")


# ### Set up the model

# Good reference on this type of model
# - Recurrent CNN: https://ojs.aaai.org/index.php/AAAI/article/view/9513/9372

# In[111]:


class CNN1dTextClassificationModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_class,
        embed_dim = 300,
        use_pretrained = True,
        fine_tune_embeddings = True
    ):

        super(CNN1dTextClassificationModel, self).__init__()

        # Set to embeddings layer of vocab_size and embed_dim vector dimension
        # Set the PADDING_IDX appropriately
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PADDING_IDX)

        if use_pretrained:
            # Set the embeddings to not requiring gradients since we'll try and modify
            self.embedding.weight.requires_grad = False
            for i in range(vocab_size):
                # Get the token for the index i
                # token = VOCAB.get_itos()[i]
                token = VOCAB.lookup_token(i) ## so much faster than get_itos
                # Modify the embedding for index i by the embedding for that token
                # Do this only if token is in the stoi dictionary for GLOVE
                if token in GLOVE.stoi:
                    self.embedding.weight[i, :] = GLOVE[token]
            # Reset to True the weights
            self.embedding.weight.requires_grad = True
        else:
            # Otherwise, initialize the weights
            self.init_weights()

        # Turn off gradients
        if not fine_tune_embeddings:
            self.embedding.weight.requires_grad = False

        # Define 3 Conv1d layers each having 1 filter and kernel sizes 2, 3 and 4
        self.cnn2 = nn.Conv1d(in_channels=embed_dim, out_channels=1, kernel_size=2)
        self.cnn3 = nn.Conv1d(in_channels=embed_dim, out_channels=1, kernel_size=3)
        self.cnn4 = nn.Conv1d(in_channels=embed_dim, out_channels=1, kernel_size=4)

        some_dim = 3  ### from error message
        self.fc = nn.Linear(in_features = some_dim, out_features=num_class)

        # For drop out + ReLu, order does not matter below
        self.dropout = nn.Dropout(p=0.3)

        self.debug = False

    def init_weights(self):
        initrange = 0.5
        # Initialize the embedding weight matrix to uniform between the [-0.5, 0.5]
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # Initialize the weight matrix of fc to uniform between the [-0.5, 0.5]
        self.fc.weight.data.uniform_(-initrange, initrange)
        # Initialize the bias for fc to zero
        self.fc.bias.data.zero_()

    # B = batch_size, L = sequence length, D = vector dimension
    def forward(self, text):

        # B X L X D
        # Get the embeddings for the text passed in
        embedded = self.embedding(text)

        if self.debug:
            print('embedding', embedded.shape)

        # B X D X L
        # Transpose the embedding above as needed
        embedded = embedded.transpose(1, 2)

        # B X 1 X L - 1
        # Pass through cnn2
        cnn2 = F.relu(self.cnn2(embedded))
        if self.debug:
            print('cnn2', cnn2.shape)

        # B X 1 X L - 2
        # Pass through cnn3
        cnn3 = F.relu(self.cnn3(embedded))
        if self.debug:
            print('cnn3', cnn3.shape)

        # B X 1 X L - 3
        # Pass through cnn4
        cnn4 = F.relu(self.cnn4(embedded))
        if self.debug:
            print('cnn4', cnn4.shape)

        # B X 1 in all cases
        # Apply max pooling to each of cnn2, cnn3 and cnn4
        cnn2 = cnn2.max(dim=2)[0]
        cnn3 = cnn3.max(dim=2)[0]
        cnn4 = cnn4.max(dim=2)[0]
        if self.debug:
            print('cnn2 after max', cnn2.shape)

        # B X 3
        # Concatenate and add drop out to the result
        cnn_concat = torch.cat((cnn2, cnn3, cnn4), dim=1)
        cnn_concat = self.dropout(cnn_concat)
        if self.debug:
            print('cnn concat', cnn_concat.shape)
            self.debug = False

        # Pass through an appropriate Linear layer to get the right dimensions needed
        out = self.fc(cnn_concat)

        return out

class RecurrentCNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_class = 4,
        e = 300, # embedding dimension
        use_pretrained = True,
        fine_tune_embeddings = True,
        # If true, this will print out the shapes of data in the forward pass for the first batch
        # This will be set to False after the first forward pass
        debug = True
    ):

        super(RecurrentCNNModel, self).__init__()

        # Set to a nn.Embedding laer for vocab_size size and e dimension
        self.embedding = nn.Embedding(vocab_size, e)

        # Set as in the paper
        self.c = 100
        self.h = 100 # hidden??
        self.initrange = 0.5

        # Same as for the CNN model above
        if use_pretrained:
            self.embedding.weight.requires_grad = False
            for i in range(vocab_size):
                # Get the token for the index i
                token = VOCAB.lookup_token(i) ## faster
                # Modify the embedding for index i by the embedding for that token
                # Do this only if token is in the stoi dictionary for GLOVE
                if token in GLOVE.stoi:
                    self.embedding.weight[i, :] = GLOVE[token]
            # Reset to True the weights
            self.embedding.weight.requires_grad = True
        else:
            # Otherwise, initialize the weights
            self.init_weights()

        if not fine_tune_embeddings:
            # Turn off gradients for the embedding weight
            self.embedding.weight.requires_grad = False

        # Set Wl, Wr, Wsl, Wsr etc as in the paper
        # Used in (1) and (2)
        ## tried using nn.Parameter but resulted in very low acc
        # self.Wl = nn.Parameter(torch.Tensor(self.c, self.c))
        self.Wl = nn.Linear(self.c,self.c)
        # self.Wr = nn.Parameter(torch.Tensor(self.c, self.c))
        self.Wr = nn.Linear(self.c, self.c)

        # Used in (1) and (2)
        self.Wsl = nn.Linear(e,self.c)
        self.Wsr =  nn.Linear(e,self.c)

        # Used in equations (4) and (6)
        # self.W2 = nn.Parameter(torch.Tensor(self.h, 2 * self.c + e))
        self.W2 = nn.Linear(e + self.c*2,self.h)
        # self.W4 = nn.Parameter(torch.Tensor(num_class, self.h))
        self.W4 = nn.Linear(self.h,num_class)

        # For drop out + ReLu, order does not matter.
        self.dropout = nn.Dropout(p=.3)

        self.debug = False

    def init_weights(self):
      # Set some of these to uniform on [-initrange, initrange]
      # The biases can be set to 0
      initrange = 0.5
      # self.embedding.weight.data.uniform_(-initrange, initrange)
      # self.Wl.weight.data.uniform_(-initrange, initrange)
      # self.Wr.weight.data.uniform_(-initrange, initrange)
      # self.Wsl.weight.data.uniform_(-initrange, initrange)
      # self.Wsr.weight.data.uniform_(-initrange, initrange)
      ## only initialize W2 and W4 to avoid the error of 'LogSoftmaxBackward0' returned nan values in its 0th output
      self.W2.weight.data.uniform_(-initrange, initrange)
      self.W4.weight.data.uniform_(-initrange, initrange)
      # self.b2 = nn.Parameter(torch.zeros(self.h))
      # self.b4 = nn.Parameter(torch.zeros(num_class))

    # B = batch_size, L = sequence length, e = vector dimension
    def forward(self, text):
        # Text is originally B X L

        # B X L X e
        embedded = self.embedding(text)

        N, L, D = embedded.shape

        # N X L X c
        cr = torch.zeros((N, L, self.c), device=text.device)
        # print(text.device == DEVICE)
        # print(text.device, DEVICE)
        if self.debug:
            print('cr ', cr.shape)
                # N X L X c

        # N X L X c
        cl = torch.zeros_like(cr)

        # N X L X c
        # We need to clone here or we get this error:
        # https://nieznanm.medium.com/runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-85d0d207623
        for l in range(1, L):
            # print(self.Wl.in_features, self.Wl.out_features, cl[:, l-1, :].transpose(0,2).shape, self.Wsl.T.shape,embedded.T.shape)
            # torch.Size([100, 100]) torch.Size([16, 60, 100]) torch.Size([300, 100]) torch.Size([16, 60, 300])
            cl[:, l, :] = F.relu(self.Wl(cl[:, l-1, :].clone()) + self.Wsl(embedded[:, l-1, :].clone()))


        # N X L X c
        # Set cr as in the paper from equation (3)
        for l in range(L-2, -1, -1):
            cr[:, l, :] = F.relu(self.Wr(cr[:, l+1, :].clone()) + self.Wsr(embedded[:, l+1, :].clone()))


        # B X L X (2c + e)
        # Set x as in the paper; this is equation (3)
        x = torch.cat([cl,embedded,cr], dim=2)
        if self.debug:
            print('x ', x.shape)

        # B X L X h
        # Set y2 as in equation (4)
        # W2: h * (e+2c) = 100 * (300 + 200),
        # print(self.W2.shape, x.shape)
        y2 = F.tanh(self.W2(x)).squeeze()
        if self.debug:
            print('y2 ', y2.shape)

        # B X H X L
        y2 = y2.permute(0, 2, 1)
        if self.debug:
            print('y2 ', y2.shape)

        # Set y3 from y2 as in equation (5)
        y3, _ = torch.max(y2, dim=2)
        # y3 = FILL
        if self.debug:
            print('y3 ', y3.shape)

        # Set y4 from W4 and y3
        # print(self.W4.shape, y3.shape)
        # y4 = (self.W4 @ y3)
        y4 = self.W4(y3)
        if self.debug:
            print('y4 ', y4.shape)
            # Set to False after this is done
            self.debug = False

        ## if self.debug, print as below:
        # torch.Size([16, 56, 100])
        # x  torch.Size([16, 56, 500])
        # y2  torch.Size([16, 56, 100])
        # y2  torch.Size([16, 100, 56])
        # y3  torch.Size([16, 100])
        # y4  torch.Size([16, 4])
        return y4


# ### Set up the model

# In[112]:


# If this is True, we will initialize the Embedding layer with GLOVE
USE_PRETRANED = True,

# If this is True, we will allow for gradient updates on the nn.Embedding layer
FINE_TUNE_EMBEDDINGS = True

# Set the loss appropriately
criterion = torch.nn.CrossEntropyLoss().to(DEVICE)


# In[113]:


# Select the Recurrent CNN Model
model = RecurrentCNNModel(len(VOCAB)).to(DEVICE)

# Set the optimizer to SGD
LR = 1.0 ### reduce learning rate to avoid the error of 'LogSoftmaxBackward0' returned nan values in its 0th output
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# Set the scheduler to StepLR with gamma=0.1 and step_size = 1.0
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size= 1)


# ### Set up the data

# In[ ]:


train_iter, test_iter = DATASETS[DATASET]()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)


# ### Train the model

# In[ ]:


def train(dataloader, model, optimizer, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 100

    for idx, (label, text) in tqdm(enumerate(dataloader)):

        optimizer.zero_grad()
        predicted_label = model(text)

        # Get the loss
        loss = criterion(predicted_label, label)

        # Do back propagation
        loss.backward()
        # if any(torch.isnan(p.grad).any() for p in model.parameters()):
        #     print("Gradient contains NaN values!")
        # Clip the gradients at 0.1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        # Do an optimization step
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += len(label)
        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(epoch, idx, len(dataloader), total_acc / total_count)
            )
            total_acc, total_count = 0, 0


# In[ ]:


def evaluate(dataloader, model):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(dataloader):
            predicted_label = model(text)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += len(label)
    return total_acc / total_count


# In[ ]:





# In[25]:


# Train the RNNCNN model
model.to(DEVICE)

## can stop when acc is acceptable
for epoch in range(1, 3):
    epoch_start_time = time.time()
    train(train_dataloader, model, optimizer, criterion, epoch)
    accu_val = evaluate(valid_dataloader, model)
    scheduler.step()
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(epoch, time.time() - epoch_start_time, accu_val)
    )
    print("-" * 59)

print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader, model)
print("test accuracy {:8.3f}".format(accu_test))


# In[21]:





# ### Train the model

# In[24]:


# Make a Conv Text model
model = CNN1dTextClassificationModel(len(VOCAB),4)

LR = 4.0 ## change LR back to 4.0

# Set the optimizer to SGD
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# Set the scheduler to StepLR with gamma=0.1 and step_size = 1.0
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size= 1)


# In[26]:


# Train the Conv1d model
model.to(DEVICE)

## can stop when acc is acceptable
for epoch in range(1, 3):
    epoch_start_time = time.time()
    train(train_dataloader, model, optimizer, criterion, epoch)
    accu_val = evaluate(valid_dataloader, model)
    scheduler.step()
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(epoch, time.time() - epoch_start_time, accu_val)
    )
    print("-" * 59)

print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader, model)
print("test accuracy {:8.3f}".format(accu_test))


# 
# ### As discussed in the paper, for text documentations, CNN only rely on fixed window size,
# ### and we cannot update such parameter to capture more semantic information
# 

# In[ ]:




