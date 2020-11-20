import torch
from torchtext import data,datasets
import torch.optim as optim
import torch.nn as nn
import random

#loading Datasets
#setting embedding to Glove vectors
TEXT=data.Field(tokenize='spacy', include_lengths = True)
LABEL=data.LabelField(dtype=torch.float)

train_data, test_data=datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data=train_data.split(random_state=random.seed(999),split_ratio=0.8)

TEXT.build_vocab(train_data,vectors="glove.6B.100d",unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, valid_iter, test_iter=data.BucketIterator.splits((train_data, valid_data, test_data),batch_size=64,sort_within_batch=True,device=device)

#embeddig dimension=100
#hidden(h_t) dimension=256
#number of layers=2
#birectional=True
class LSTM(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size, 100, padding_idx=pad_idx)
        self.rnn=nn.LSTM(100,256,num_layers=2,bidirectional=True,dropout=0.5)
        self.fc=nn.Linear(512,1)
        self.dropout=nn.Dropout(0.5)
        
    def forward(self, text, length):
        embedded=self.dropout(self.embedding(text))
        packed_embeding=nn.utils.rnn.pack_padded_sequence(embedded, length.cpu()) #Packing
        output, (hidden, cell)=self.rnn(packed_embeding)
        output, output_lengths=nn.utils.rnn.pad_packed_sequence(output) #unpacking
        hidden=self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

model=LSTM(len(TEXT.vocab),TEXT.vocab.stoi[TEXT.pad_token])

#copying mebedding weights from glove vectors
model.embedding.weight.data.copy_(TEXT.vocab.vectors)

model.embedding.weight.data[TEXT.vocab.stoi[TEXT.unk_token]]=torch.zeros(100)
model.embedding.weight.data[TEXT.vocab.stoi[TEXT.pad_token]]=torch.zeros(100)
model=model.to(device)

#setting loss function and optimizer
optimizer=optim.Adam(model.parameters())
criterion=nn.BCEWithLogitsLoss()
criterion=criterion.to(device)

def accuracy(pred, y):
    model_pred=torch.round(torch.sigmoid(pred))
    correct=(model_pred==y).int()
    acc=correct.sum()/len(correct)
    return acc

def train(model, iter, optimizer, criterion):
    total_loss=0
    total_acc=0
    model.train()
    for batch in iter:
        optimizer.zero_grad()
        text, text_lengths=batch.text
        predictions=model(text, text_lengths).squeeze(1)
        loss=criterion(predictions, batch.label)
        acc=accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        total_acc+=acc.item()
    return total_loss / len(iter), total_acc / len(iter)

def evaluate(model, iter, criterion):
    total_loss=0
    total_acc=0
    model.eval()
    with torch.no_grad():
        for batch in iter:
            text, text_lengths=batch.text
            predictions=model(text, text_lengths).squeeze(1)
            loss=criterion(predictions, batch.label)
            acc=accuracy(predictions, batch.label)
            total_loss+=loss.item()
            total_acc+=acc.item()
    return total_loss / len(iter), total_acc / len(iter)

#training model
epochs=20
best_loss=float('inf')
for epoch in range(epochs):
    train_loss, train_acc=train(model, train_iter, optimizer, criterion)
    valid_loss, valid_acc=evaluate(model, valid_iter, criterion)
    if valid_loss<best_loss:
        best_loss=valid_loss
        torch.save(model.state_dict(), 'Bidirectional_LSTM.pt')
    
    print(f'Epoch: {epoch+1}   Train Loss: {train_loss:.4f}   Train Accuracy: {train_acc*100:.2f}%   Valiation Loss: {valid_loss:.4f}   Validtion Accuracy: {valid_acc*100:.2f}%')

#testing model
model.load_state_dict(torch.load('Bidirectional_LSTM.pt'))
test_loss, test_acc = evaluate(model, test_iter, criterion)
print(f'Test Loss: {test_loss:.4f}   Test Acc: {test_acc*100:.2f}%')
