import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset

class ADRDNN(nn.Module):
	def __init__(self):
		super(ADRDNN, self).__init__()

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

# The same !


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


test_data = pd.read_csv('../Data/test_data.csv').to_numpy().astype(np.float64)

test_timestamp = test_data[:,0:3]
test_day_info = np.sum(test_data[:,3:5], axis=1)

N = test_day_info.shape[0]
test_day_info = test_day_info.reshape(N, 1)

test_data = test_data[:,3:]

X = torch.Tensor(test_data)
adr_pred = np.zeros((N, 1))
xrate_pred = np.zeros((N, 1))

NofM = 10


for nm in range(NofM):

	adr_model_path = '../Model/adr_model_{}.mdl'.format(nm)
	adr_model = ADRDNN().cuda()

	xrate_model_path = '../Model/xrate_model_{}.mdl'.format(nm)
	xrate_model = XRATEDNN().cuda()

	try:
		adr_model.load_state_dict(torch.load(adr_model_path))
	except:
		print('Read no model')

	try:
		xrate_model.load_state_dict(torch.load(xrate_model_path))
	except:
		print('Read no model')

	adr_model.eval()
	xrate_model.eval()


	with torch.no_grad():
		adr_pred += adr_model(X.cuda()).cpu().detach().numpy()
		xrate_pred += xrate_model(X.cuda()).cpu().detach().numpy()

adr_pred /= NofM
xrate_pred /= NofM

print(test_timestamp.shape)
print(test_day_info.shape)
print(adr_pred.shape)
print(xrate_pred.shape)

print(xrate_pred)

xrate_pred[xrate_pred >= 1] = 1

revenue = test_day_info * adr_pred * (1 - xrate_pred)

combined_data = np.concatenate((test_timestamp, revenue), axis=1)

day_info, spliter = np.unique(test_timestamp, axis=0, return_index=True)

true_revenue = np.array([np.sum(x) for x in np.split(revenue, spliter)[1:]])

true_revenue /= (np.max(true_revenue) + 1)

answer = (true_revenue * 10).astype(np.int32)

with open('../Result/answer.csv', 'w') as f:
	f.write('arrival_date,label\n')
	for i, l in enumerate(answer):
		y,m,d = day_info[i].astype(np.int32)
		f.write('{}-{:02d}-{:02d},{}\n'.format(y,m,d,l))