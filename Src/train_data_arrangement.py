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

train = train[['is_canceled', 'arrival_date_year','arrival_date_month', 'arrival_date_day_of_month', 
'hotel', 'is_repeated_guest', 'lead_time', 'stays_in_weekend_nights',
'stays_in_week_nights', 'adults', 'children', 'babies',
'total_of_special_requests', 'reserved_room_type', 'customer_type', 'meal', 'country']]

train['hotel'] = train['hotel'].replace({'Resort Hotel' : 0, 'City Hotel' : 1})
train['country'] = train['country'].replace({None : 'NP'})
train['meal'] = train['meal'].replace({'Undefined' : 'NN'})

train = train.rename(columns=
	{ 'arrival_date_year' : 'Y', 
	  'arrival_date_month': 'M',
	  'arrival_date_day_of_month':'D',
	  'hotel' : 'HotelType',
	  'is_repeated_guest' : 'R',
	  'is_canceled' : 'X',
	  'total_of_special_requests' : 'TotalSR',
	  'lead_time' : 'LT',
	  'reserved_room_type' : 'RoomType',
	  'customer_type' : 'GuestType',
	  'stays_in_weekend_nights' : 'WeekendNight',
	  'stays_in_week_nights' : 'WeekNight',
	  'adults' : 'A', 'children' : 'C', 'babies' : 'B'})


cns = sorted(list(set(train['M'])))
print(cns)

pending_lists = [list() for cn in cns]

for i, cn in enumerate(cns):
	for r in train['M']:
		pending_lists[i].append(int(r == cn))

for i, cn in enumerate(cns):
	train['MonthIs{}'.format(cn)] = pending_lists[i]

del cns
del pending_lists

cns = list(set(train['country']))

for cn in cns:
	s = len(train[train['country'] == cn])
	if s <= 1500:
		train['country'] = train['country'].replace({cn : 'NP'})

cns = sorted(list(set(train['country'])))

pending_lists = [list() for cn in cns]

for i, cn in enumerate(cns):
	for r in train['country']:
		pending_lists[i].append(int(r == cn))

for i, cn in enumerate(cns):
	train['CountryIs{}'.format(cn)] = pending_lists[i]

del train['country']
del cns
del pending_lists

cns = sorted(list(set(train['RoomType'])))

pending_lists = [list() for cn in cns]

for i, cn in enumerate(cns):
	for r in train['RoomType']:
		pending_lists[i].append(int(r == cn))

for i, cn in enumerate(cns):
	train['RoomTypeIs{}'.format(cn)] = pending_lists[i]

del train['RoomType']
del cns
del pending_lists

cns = sorted(list(set(train['GuestType'])))

pending_lists = [list() for cn in cns]

for i, cn in enumerate(cns):
	for r in train['GuestType']:
		pending_lists[i].append(int(r == cn))

for i, cn in enumerate(cns):
	train['GuestTypeIs{}'.format(cn)] = pending_lists[i]

del train['GuestType']
del cns
del pending_lists

cns = sorted(list(set(train['meal'])))

pending_lists = [list() for cn in cns]

for i, cn in enumerate(cns):
	for r in train['meal']:
		pending_lists[i].append(int(r == cn))

for i, cn in enumerate(cns):
	train['MealIs{}'.format(cn)] = pending_lists[i]

del train['meal']
del cns
del pending_lists

group = train.groupby(by=['Y','M','D'])


train = pd.concat((group.size(), group.mean()), axis=1)

with open('../Data/train_data.csv', 'w') as f:
	train.to_csv(f, index=True, float_format='%.5f')

train_label = open('../RawData/train_label.csv', 'r').read().split('\n')

train_label[0] = 'Y,M,D,label'

for i, line in enumerate(train_label[1:]):
	if line == '': continue
	date, label = line.split(',')
	y, m, d = date.split('-')
	y = str(int(y))
	m = str(int(m))
	d = str(int(d))

	train_label[i+1] = '{},{},{},{}'.format(y,m,d,label)

open('../Data/train_label.csv', 'w').write('\n'.join(train_label))