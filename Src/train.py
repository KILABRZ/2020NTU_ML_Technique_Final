import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch import nn

class DNN(nn.Module):
	def __init__(self):
		super(DNN, self).__init__()

		self.dnn = nn.Sequential(
			nn.Flatten(),
			nn.Linear(52, 52),
			nn.ReLU(),
			nn.Linear(52, 52),
			nn.ReLU(),
			nn.Linear(52, 52),
			nn.ReLU(),
			nn.Linear(52, 52),
			nn.ReLU(),
			nn.Linear(52, 1)
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
train_label = pd.read_csv('../Data/train_label.csv').to_numpy().astype(np.float64)

train_data = train_data[:,1:]
train_data[:,2] *= (1 - train_data[:,3])
train_data = np.concatenate((train_data[:,0:3], train_data[:,4:]), axis=1)
train_data = train_data[:,2:]

train_label = train_label[:,3:4]

X = torch.Tensor(train_data)
Y = torch.Tensor(train_label)


model = DNN().cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.05)
loss_function = nn.L1Loss()

T = 100000
train_val_ratio = 0.75

best_loss = 999999

train_X, train_Y, val_X, val_Y = RandomDataSplit(X, Y, train_val_ratio)

train_X, train_Y, val_X, val_Y = \
train_X.cuda(), train_Y.cuda(), val_X.cuda(), val_Y.cuda()

for epoch in range(T):

	print('[{}/{}] training... '.format(epoch+1, T), end='\r')



	model.train()
	pred = model(train_X)
	loss = loss_function(pred, train_Y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	model.eval()
	with torch.no_grad():
		val_pred = model(val_X).int()
		loss = loss_function(val_pred, val_Y).item()
		print('Validation loss = {:.5f}         '.format(loss), end='\r')

		if loss < best_loss :
			print('Update model with better loss = {:.5f}'.format(loss))
			best_loss = loss
			model_path = '../Model/dnn_model.mdl'
			torch.save(model.state_dict(), model_path)


