import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torchtext import data,datasets

#!pip install transformers
from transformers import BertTokenizer,BertModel

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

init_index=tokenizer.cls_token_id
eos_index=tokenizer.sep_token_id 

max_length=tokenizer.max_model_input_sizes['bert-base-uncased']

def tokenize_input(sentence):
    tokens=tokenizer.tokenize(sentence) 
    tokens=tokens[:max_length-2]
    return tokens

TEXT=data.Field(batch_first=True,use_vocab = False,tokenize = tokenize_input,preprocessing = tokenizer.convert_tokens_to_ids,init_token = init_index,eos_token = eos_index,pad_token = tokenizer.pad_token_id,unk_token = tokenizer.unk_token_id)
LABEL=data.LabelField(dtype=torch.float)
LABEL.build_vocab(train_data)

train_data, test_data=datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data=train_data.split(random_state=random.seed(999),split_ratio=0.8)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),batch_size = 64,device = device)

bert = BertModel.from_pretrained('bert-base-uncased')

#Hidden_dimension=256
#number of layers=2
#Bidirectional model
#dropout=0.25 
#accuracy reduces with dropout=0.5
class GRU(nn.Module):
    def __init__(self,bert):
        super().__init__()
        self.bert=bert
        embedding_dim=bert.config.to_dict()['hidden_size'] #786
        self.rnn=nn.GRU(embedding_dim,256,num_layers = 2,bidirectional =True,batch_first = True,dropout = 0.25)
        self.fc=nn.Linear(512,1)
        self.dropout=nn.Dropout(0.25)
    def forward(self, text):                
        with torch.no_grad():
            embedded=self.bert(text)[0]
        output, hidden=self.rnn(embedded)
        hidden=self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

model=GRU(bert)
model = model.to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

def train(model, iter, optimizer, criterion):
    total_loss = 0
    total_acc = 0
    model.train()
    for batch in iter:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc.item()
    return total_loss / len(iter), total_acc / len(iter)

def accuracy(preds, y):
    model_pred=torch.round(torch.sigmoid(preds))
    correct=(model_pred==y).int()
    acc=correct.sum() / len(correct)
    return acc

def evaluate(model, iter, criterion):    
    total_loss=0
    total_acc=0
    model.eval()
    with torch.no_grad():
        for batch in iter:
            predictions=model(batch.text).squeeze(1)
            loss=criterion(predictions, batch.label)
            acc=accuracy(predictions, batch.label)
            total_loss+=loss.item()
            total_acc+=acc.item()
    return total_loss/len(iter), total_acc/len(iter)

best_loss=float('inf')
epochs=5
for epoch in range(epochs):
    train_loss, train_acc=train(model, train_iter, optimizer, criterion)
    valid_loss, valid_acc=evaluate(model, valid_iter, criterion)
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), 'Transformer.pt')
    
    print(f'Epoch: {epoch+1}   Train Loss: {train_loss:.4f}   Train Accuracy: {train_acc*100:.2f}%   Validation Loss: {valid_loss:.3f} |  Validation Accuracy: {valid_acc*100:.2f}%')
#every epoch takes appx 20 minutes to train :(

model.load_state_dict(torch.load('Transformer.pt'))
test_loss, test_acc = evaluate(model, test_iter, criterion)
print(f'Test Loss: {test_loss:.4f}   Test Acc: {test_acc*100:.2f}%')

#from google.colab import drive
#drive.mount('/content/gdrive')
#model_save_name = 'Transformer.pt'
#path = F"/content/gdrive/My Drive/Colab Notebooks/{model_save_name}" 
#torch.save(model.state_dict(), path)
