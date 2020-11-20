import torch
from torchtext import data,datasets
import random
import torch.nn as nn
#Loading data, Tokenizing it, and splitting it into training, validation and test data
TEXT=data.Field(tokenize='spacy')
LABEL=data.LabelField(dtype=torch.float)
train_data, test_data=datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data=train_data.split(random_state=random.seed(999),split_ratio=0.8)
TEXT.build_vocab(train_data)
LABEL.build_vocab(train_data)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, valid_iter, test_iter=data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=64, device=device)

#Defining model
# if want LSTM comment line 27,35 and uncomment line 28,36
# if RNN,LSTM is multilayered and not bidirectional then uncomment line 37
# if RNN, LSTM is bidirectional id bidirectinal then uncomment line 38
#Embedding Dimension=200
#Hidden DImension (h_t)=256
class RNN(nn.Module):
    def __init__(self, input_dim,num_layers,bidirectional):
        super().__init__()
        self.embedding=nn.Embedding(input_dim,200)
        self.rnn=nn.RNN(200,256,num_layers,bidirectional)
        #self.rnn=nn.LSTM(200,256,num_layers,bidirectional)
        self.fc1=nn.Linear(num_layers*256,64)
        self.fc2=nn.Linear(64,1)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, text):
        embedded=self.embedding(text)
        output, hidden=self.rnn(embedded)
        #output, (hidden,cell)=self.LSTM(embedded)
        #hidden=hidden[-1,:,:]
        #hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        out=self.fc1(hidden.squeeze(0))
        out=self.sigmoid(out)
        return self.fc2(out) 


#For Multilayer RNN, LSTM :  set num_layers to that value
#If want bidirectional RNN, LSTM set bidirectional=True
num_layers=1
bidirectional=False
model=RNN(len(TEXT.vocab),num_layers,bidirectional)
model=model.to(device)

#Choosing Optimizer and Loss
optimizer=torch.optim.SGD(model.parameters(), lr=3e-5)

criterion=nn.BCEWithLogitsLoss()
criterion=criterion.to(device)

def accuracy(preds, y):
    model_preds=torch.round(torch.sigmoid(preds))
    correct=(model_preds==y).int()
    acc=correct.sum()/len(correct)
    return acc

def train(model, iter, optimizer, criterion):
    total_loss=0
    total_acc=0
    model.train()
    for batch in iter:
        optimizer.zero_grad()  
        predictions=model(batch.text).squeeze(1)
        loss=criterion(predictions, batch.label)
        acc=accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        total_acc+=acc.item()
    return total_loss/len(iter), total_acc/len(iter)

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

#Training Model
epochs=20
best_loss = float('inf')
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), 'RNN_model.pt')
    print(f'Epoch: {epoch+1}   Train Loss: {train_loss:.4f}   Train Acc: {train_acc*100:.2f}%   Val. Loss: {valid_loss:.4f}   Val. Acc: {valid_acc*100:.2f}%')

#Testing model
model.load_state_dict(torch.load('RNN_model.pt'))
test_loss, test_acc = evaluate(model, test_iter, criterion)
print(f'Test Loss: {test_loss:.4f}   Test Acc: {test_acc*100:.2f}%')
