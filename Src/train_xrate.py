import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

class NormalDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def __len__(self):
		return len(self.x)
	def __getitem__(self, index):
		x = self.x[index]
		y = self.y[index]
		return x, y

class XRATEDNN(nn.Module):
	def __init__(self):
		super(XRATEDNN, self).__init__()

		D = 80
		self.dnn = nn.Sequential(
			nn.Flatten(),
			nn.Linear(D, D),
			nn.ReLU(),
			nn.Linear(D, D),
			nn.ReLU(),
			nn.Linear(D, D),
			nn.ReLU(),
			nn.Linear(D, D),
			nn.ReLU(),
			nn.Linear(D, 1)
		)

	def forward(self, x):
		return self.dnn(x)

def RandomDataSplit(X, Y, r):
	N = X.shape[0]
	s = np.random.permutation(N)

	TBD = int(N * r)

	train_part = s[:TBD]
	val_part = s[TBD:]

	train_X = X[train_part]
	train_Y = Y[train_part]
	val_X = X[val_part]
	val_y = Y[val_part]
	return train_X, train_Y, val_X, val_y


train_data = pd.read_csv('../Data/train_data.csv').to_numpy().astype(np.float64)
train_label = pd.read_csv('../Data/xrate_train_label.csv').to_numpy().astype(np.float64)

train_data = train_data[:,3:]

X = torch.Tensor(train_data)
Y = torch.Tensor(train_label)

model_path = '../Model/xrate_model.mdl'
model = XRATEDNN().cuda()
try:
	model.load_state_dict(torch.load(model_path))
except:
	pass
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.001)
loss_function = nn.MSELoss()

T = 5000

train_val_ratio = 0.8

best_loss = 999999

train_X, train_Y, val_X, val_Y = RandomDataSplit(X, Y, train_val_ratio)

train_set = NormalDataset(train_X, train_Y)
val_set = NormalDataset(val_X, val_Y)

train_len = len(train_set)
val_len = len(val_set)

Bsize = 32768

train_loader = DataLoader(train_set, batch_size = Bsize, shuffle = True)
val_loader = DataLoader(val_set, batch_size = Bsize, shuffle = False)



for epoch in range(T):

	print('[{}/{}] training... '.format(epoch+1, T), end='')

	model.train()

	train_loss = 0
	val_loss = 0

	train_acc = 0
	val_acc = 0

	for i, data in enumerate(train_loader):
		x, y = data
		pred = model(x.cuda())
		if True in torch.isnan(pred):
			print('explode')
			print(i)
			print(x)
			print(y)
			exit()
		loss = loss_function(pred, y.cuda())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss += loss.item() * x.shape[0]

	model.eval()
	
	with torch.no_grad():
		for i, data in enumerate(val_loader):
			x, y = data
			pred = model(x.cuda())
			loss = loss_function(pred, y.cuda()).item()
			val_loss += loss * x.shape[0]

	train_loss /= train_len
	val_loss /= val_len
	print('Validation Loss = {:.5f}  Train Loss = {:.5f}'.format(val_loss, train_loss), end='\r')

	if val_loss < best_loss :
		print('\nUpdate model with better loss = {:.5f}'.format(val_loss))
		best_loss = val_loss
		torch.save(model.state_dict(), model_path)


