# Arrange train data to the type of (features), (label) to enable next step of works
import pandas as pd
from tqdm import tqdm

train = pd.read_csv('../RawData/train.csv')
train = train.sort_values(by=['arrival_date_day_of_month'])
transform = {'January' : 1, 'February' : 2, 'March' : 3, 'April' : 4, 'May' : 5, 'June' : 6,
'July' : 7, 'August' : 8, 'September' : 9, 'October' : 10, 'November' : 11, 'December' : 12}
train['arrival_date_month'] = train['arrival_date_month'].replace(transform)
del train['ID']
train = train.sort_values(by=['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'], kind='mergesort')

train_label = open('../RawData/train_label.csv', 'r').read().split('\n')[1:]
quest_table = dict()

for line in train_label:
	if line == '': break
	date, point = line.split(',')
	year, month, day = date.split('-')
	year, month, day, point = int(year), int(month), int(day), float(point)
	quest_table[(year, month, day)] = point

pending_list = list()

for row in tqdm(train.iterrows()):
	year, month, day = row[1]['arrival_date_year'], row[1]['arrival_date_month'], row[1]['arrival_date_day_of_month']
	tup = (year, month, day)
	if quest_table.get(tup, None) == None:
		pending_list.append(-5)
	else:
		pending_list.append(quest_table[tup])

train['revenue_scale'] = pending_list
train = train[train.revenue_scale >= 0]

with open('../Data/first_train_data.csv', 'w') as f:
	train.to_csv(f, index=False)