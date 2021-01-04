# Arrange test data to the type of (features), (label) to enable next step of works
import pandas as pd
from tqdm import tqdm

test = pd.read_csv('../RawData/test.csv')
test = test.sort_values(by=['arrival_date_day_of_month'])
transform = {'January' : 1, 'February' : 2, 'March' : 3, 'April' : 4, 'May' : 5, 'June' : 6,
'July' : 7, 'August' : 8, 'September' : 9, 'October' : 10, 'November' : 11, 'December' : 12}
test['arrival_date_month'] = test['arrival_date_month'].replace(transform)
del test['ID']
test = test.sort_values(by=['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'], kind='mergesort')

test = test[['arrival_date_year','arrival_date_month', 'arrival_date_day_of_month', 
'hotel', 'is_repeated_guest', 'lead_time', 'stays_in_weekend_nights',
'stays_in_week_nights', 'adults', 'children', 'babies',
'total_of_special_requests', 'reserved_room_type', 'customer_type', 'meal', 'country']]

test['hotel'] = test['hotel'].replace({'Resort Hotel' : 0, 'City Hotel' : 1})
test['country'] = test['country'].replace({None : 'NP'})
test['meal'] = test['meal'].replace({'Undefined' : 'NN'})

test = test.rename(columns=
	{ 'arrival_date_year' : 'Y', 
	  'arrival_date_month': 'M',
	  'arrival_date_day_of_month':'D',
	  'hotel' : 'HotelType',
	  'is_repeated_guest' : 'R',
	  'total_of_special_requests' : 'TotalSR',
	  'lead_time' : 'LT',
	  'reserved_room_type' : 'RoomType',
	  'customer_type' : 'GuestType',
	  'stays_in_weekend_nights' : 'WeekendNight',
	  'stays_in_week_nights' : 'WeekNight',
	  'adults' : 'A', 'children' : 'C', 'babies' : 'B'})

cns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

pending_lists = [list() for cn in cns]

for i, cn in enumerate(cns):
	for r in test['M']:
		pending_lists[i].append(int(r == cn))

for i, cn in enumerate(cns):
	test['MonthIs{}'.format(cn)] = pending_lists[i]

del cns
del pending_lists


cns = list(set(test['country']))
true_cns = ['FRA', 'NLD', 'NP', 'BEL', 'BRA', 'ITA', 'DEU', 'ESP', 'PRT', 'GBR', 'IRL']

for cn in cns:
	if cn not in true_cns:
		test['country'] = test['country'].replace({cn : 'NP'})

cns = sorted(['FRA', 'NLD', 'NP', 'BEL', 'BRA', 'ITA', 'DEU', 'ESP', 'PRT', 'GBR', 'IRL'])

pending_lists = [list() for cn in cns]

for i, cn in enumerate(cns):
	for r in test['country']:
		pending_lists[i].append(int(r == cn))

for i, cn in enumerate(cns):
	test['CountryIs{}'.format(cn)] = pending_lists[i]

del test['country']
del cns
del pending_lists



cns = list(set(test['RoomType']))
true_cns = ['B', 'A', 'H', 'C', 'L', 'P', 'D', 'G', 'E', 'F']

for cn in cns:
	if cn not in true_cns:
		test['RoomType'] = test['RoomType'].replace({cn : 'B'})

cns = sorted(['B', 'A', 'H', 'C', 'L', 'P', 'D', 'G', 'E', 'F'])

pending_lists = [list() for cn in cns]

for i, cn in enumerate(cns):
	for r in test['RoomType']:
		pending_lists[i].append(int(r == cn))

for i, cn in enumerate(cns):
	test['RoomTypeIs{}'.format(cn)] = pending_lists[i]

del test['RoomType']
del cns
del pending_lists

cns = list(set(test['GuestType']))
true_cns = ['Transient-Party', 'Contract', 'Group', 'Transient']

for cn in cns:
	if cn not in true_cns:
		test['GuestType'] = test['GuestType'].replace({cn : 'Transient'})

cns = sorted(['Transient-Party', 'Contract', 'Group', 'Transient'])

pending_lists = [list() for cn in cns]

for i, cn in enumerate(cns):
	for r in test['GuestType']:
		pending_lists[i].append(int(r == cn))

for i, cn in enumerate(cns):
	test['GuestTypeIs{}'.format(cn)] = pending_lists[i]

del test['GuestType']
del cns
del pending_lists

cns = list(set(test['meal']))
true_cns = ['HB', 'FB', 'BB', 'NN', 'SC']

for cn in cns:
	if cn not in true_cns:
		test['meal'] = test['meal'].replace({cn : 'NN'})

cns = sorted(['HB', 'FB', 'BB', 'NN', 'SC'])

pending_lists = [list() for cn in cns]

for i, cn in enumerate(cns):
	for r in test['meal']:
		pending_lists[i].append(int(r == cn))

for i, cn in enumerate(cns):
	test['MealIs{}'.format(cn)] = pending_lists[i]

del test['meal']
del cns
del pending_lists

group = test.groupby(by=['Y','M','D'])


test = pd.concat((group.size(), group.mean()), axis=1)

with open('../Data/test_data.csv', 'w') as f:
	test.to_csv(f, index=True, float_format='%.5f')

test_label = open('../RawData/test_nolabel.csv', 'r').read().split('\n')

test_label[0] = 'Y,M,D'

for i, line in enumerate(test_label[1:]):
	if line == '': continue
	y, m, d = line.split('-')
	y = str(int(y))
	m = str(int(m))
	d = str(int(d))

	test_label[i+1] = '{},{},{}'.format(y,m,d)

open('../Data/test_nolabel.csv', 'w').write('\n'.join(test_label))