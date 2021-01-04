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
			nn.Linear(42, 42),
			nn.ReLU(),
			nn.Linear(42, 42),
			nn.ReLU(),
			nn.Linear(42, 42),
			nn.ReLU(),
			nn.Linear(42, 42),
			nn.ReLU(),
			nn.Linear(42, 1)
		)

	def forward(self, x):
		return self.dnn(x)

model_path = '../Model/dnn_model.mdl'
model = DNN().cuda()
model.load_state_dict(torch.load(model_path))


test_data = pd.read_csv('../Data/test_data.csv').to_numpy().astype(np.float64)
test_timestamp = test_data[:,:3].astype(np.int32)
test_data = test_data[:,1:]
test_data = torch.Tensor(test_data)


model.eval()
pred = model(test_data.cuda()).int().cpu().detach().numpy()

answer = np.concatenate((test_timestamp, pred), axis=1)
answer_sheet = ['{}-{:02d}-{:02d},{}'.format(a, b, c, d) for a, b, c, d in answer]

with open('../Result/answer.csv', 'w') as fp:
	fp.write('arrival_date,label\n')
	fp.write('\n'.join(answer_sheet))