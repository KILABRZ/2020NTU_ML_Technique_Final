import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier

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

test_data = pd.read_csv('../Data/test_data.csv').to_numpy().astype(np.float64)

test_timestamp = test_data[:,0:3]
test_day_info = np.sum(test_data[:,3:5], axis=1)

N = test_day_info.shape[0]
test_day_info = test_day_info.reshape(N, 1)

test_data = test_data[:,3:]

train_data = pd.read_csv('../Data/train_data.csv').to_numpy().astype(np.float64)
train_label = pd.read_csv('../Data/xrate_train_label.csv').to_numpy().astype(np.float64)
train_data = train_data[:,3:]
tX = train_data
tY = train_label.reshape(-1)

print(tX.shape)
print(tY.shape)

X = torch.Tensor(test_data)
adr_model_path = '../Model/adr_model.mdl'
adr_model = ADRDNN().cuda()

try:
	adr_model.load_state_dict(torch.load(adr_model_path))
except:
	print('Read no model')

adr_model.eval()

with torch.no_grad():
	adr_pred = adr_model(X.cuda()).cpu().detach().numpy()


forest = RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(tX, tY)
xrate_pred = forest.predict(X)

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