import pandas as pd
import numpy as np

test_data = pd.read_csv('../RawData/test.csv')

test_data['arrival_date_month'] = test_data['arrival_date_month'].replace({
	'January' : 1, 'February' : 2, 'March' : 3, 'April' : 4, 'May' : 5,
	'June' : 6, 'July' : 7, 'August' : 8, 'September' : 9, 'October' : 10,
	'November' : 11, 'December' : 12
})
test_data = test_data.sort_values(by=['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'])

useful_field = ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']
useful_field += ['stays_in_weekend_nights', 'stays_in_week_nights']
useful_field += ['adults', 'children', 'babies', 'is_repeated_guest']
useful_field += ['meal', 'country', 'market_segment']
useful_field += ['hotel', 'reserved_room_type', 'assigned_room_type', 'customer_type']
useful_field += ['required_car_parking_spaces', 'total_of_special_requests']

test_data = test_data[useful_field]

# deal with marking data

# meal
# print(pd.unique(test_data['meal']))

test_data['meal'] = test_data['meal'].replace({'Undefined' : 'UN'})
mark_list = ['BB', 'HB', 'FB', 'SC', 'UN']
pending_lists = [list() for _ in mark_list]

for i, mark in enumerate(mark_list):
	for thing in test_data['meal']:
		if thing == mark:
			pending_lists[i].append(1)
		else:
			pending_lists[i].append(0)

for i, mark in enumerate(mark_list):
	test_data['meal_is_{}'.format(mark)] = pending_lists[i]

del mark_list
del pending_lists
del test_data['meal']

# country
# pd.unique(test_data['country'])
test_data['country'] = test_data['country'].fillna('Unknown')
mark_list = ['PRT', 'IRL', 'GBR', 'ESP', 'USA', 'FRA', 'ARG', 'OMN', 'NOR', 'DEU', 'ROU',
 'POL', 'ITA', 'BRA', 'BEL', 'CHE', 'CN', 'NLD', 'GRC', 'DNK', 'SWE', 'RUS', 'EST',
 'AUS', 'CZE', 'FIN', 'AUT', 'ISR', 'HUN', 'MOZ', 'BWA', 'NZL', 'LUX', 'IDN', 'SVN',
 'ALB', 'MAR', 'HRV', 'CHN', 'AGO', 'BGR', 'IND', 'DZA', 'MEX', 'TUN', 'COL', 'KAZ',
 'LVA', 'STP', 'UKR', 'VEN', 'TWN', 'IRN', 'SMR', 'TUR', 'KOR', 'BLR', 'JPN', 'PRI',
 'SRB', 'LTU', 'CPV', 'AZE', 'LBN', 'CRI', 'CHL', 'THA', 'SVK', 'CMR', 'EGY', 'LIE',
 'MYS', 'SAU', 'ZAF', 'MKD', 'MMR', 'DOM', 'IRQ', 'SGP', 'CYM', 'ZMB', 'PAN', 'ZWE',
 'SEN', 'NGA', 'GIB', 'ARM', 'PER', 'LKA', 'KWT', 'JOR', 'KNA', 'GEO', 'TMP', 'ETH',
 'ECU', 'MUS', 'PHL', 'CUB', 'ARE', 'BFA', 'AND', 'CYP', 'KEN', 'BIH', 'COM', 'SUR',
 'JAM', 'HND', 'MCO', 'GNB', 'LBY', 'RWA', 'PAK', 'UGA', 'TZA', 'CIV', 'SYR', 'QAT',
 'KHM', 'HKG', 'BGD', 'MLI', 'ISL', 'UZB', 'BHR', 'URY', 'NAM', 'BOL', 'IMN', 'BDI',
 'TJK', 'MLT', 'MDV', 'NIC', 'SYC', 'PRY', 'BRB', 'ABW', 'GGY', 'AIA', 'VNM', 'SLV',
 'PLW', 'BEN', 'MAC', 'DMA', 'VGB', 'JEY', 'GAB', 'PYF', 'CAF', 'LCA', 'GUY', 'ATA',
 'GHA', 'MWI', 'MNE', 'GLP', 'GTM', 'MDG', 'ASM', 'TGO', 'Unknown']

pending_lists = [list() for _ in mark_list]

for i, mark in enumerate(mark_list):
	s = len(test_data[test_data['country'] == mark])
	if s < 1000:
		test_data['country'] = test_data['country'].replace({mark : 'Unknown'})

mark_list = ['PRT', 'IRL', 'GBR', 'ESP', 'USA',
'FRA', 'DEU', 'ITA', 'BRA', 'BEL', 'CHE', 'NLD', 'Unknown']

for i, mark in enumerate(mark_list):
	for thing in test_data['country']:
		if thing == mark:
			pending_lists[i].append(1)
		else:
			pending_lists[i].append(0)

for i, mark in enumerate(mark_list):
	test_data['country_is_{}'.format(mark)] = pending_lists[i]

del mark_list
del pending_lists
del test_data['country']

# market_segment
# print(list(pd.unique(test_data['market_segment'])))

test_data['market_segment'] = test_data['market_segment'].replace({'Undefined' : 'Unknown'})

mark_list = ['Direct', 'Offline TA/TO', 'Online TA', 'Corporate',
'Groups', 'Complementary', 'Aviation', 'Unknown']

pending_lists = [list() for _ in mark_list]

for i, mark in enumerate(mark_list):
	for thing in test_data['market_segment']:
		if thing == mark:
			pending_lists[i].append(1)
		else:
			pending_lists[i].append(0)

for i, mark in enumerate(mark_list):
	test_data['market_segment_is_{}'.format(mark)] = pending_lists[i]

del mark_list
del pending_lists
del test_data['market_segment']

# reserved_room_type
# print(list(pd.unique(test_data['assigned_room_type'])))
mark_list = ['C', 'A', 'E', 'B', 'D', 'I', 'F', 'G', 'H', 'L', 'K', 'P', 'Unknown']

pending_lists = [list() for _ in mark_list]

for i, mark in enumerate(mark_list):
	for thing in test_data['reserved_room_type']:
		if thing == mark:
			pending_lists[i].append(1)
		else:
			pending_lists[i].append(0)

for i, mark in enumerate(mark_list):
	test_data['reserved_room_type_is_{}'.format(mark)] = pending_lists[i]

del mark_list
del pending_lists
del test_data['reserved_room_type']

# assigned_room_type
# print(list(pd.unique(test_data['assigned_room_type'])))
mark_list = ['C', 'A', 'E', 'B', 'D', 'I', 'F', 'G', 'H', 'L', 'K', 'P', 'Unknown']

pending_lists = [list() for _ in mark_list]

for i, mark in enumerate(mark_list):
	for thing in test_data['assigned_room_type']:
		if thing == mark:
			pending_lists[i].append(1)
		else:
			pending_lists[i].append(0)

for i, mark in enumerate(mark_list):
	test_data['assigned_room_type_is_{}'.format(mark)] = pending_lists[i]

del mark_list
del pending_lists
del test_data['assigned_room_type']

# hotel
# print(list(pd.unique(test_data['assigned_room_type'])))
mark_list = ['City Hotel', 'Resort Hotel', 'Unknown']

pending_lists = [list() for _ in mark_list]

for i, mark in enumerate(mark_list):
	for thing in test_data['hotel']:
		if thing == mark:
			pending_lists[i].append(1)
		else:
			pending_lists[i].append(0)

for i, mark in enumerate(mark_list):
	test_data['hotel_is_{}'.format(mark)] = pending_lists[i]

del mark_list
del pending_lists
del test_data['hotel']

# customer_type
# print(list(pd.unique(test_data['customer_type'])))
mark_list = ['Transient', 'Transient-Party', 'Contract', 'Group', 'Unknown']

pending_lists = [list() for _ in mark_list]

for i, mark in enumerate(mark_list):
	for thing in test_data['customer_type']:
		if thing == mark:
			pending_lists[i].append(1)
		else:
			pending_lists[i].append(0)

for i, mark in enumerate(mark_list):
	test_data['customer_type_is_{}'.format(mark)] = pending_lists[i]

del mark_list
del pending_lists
del test_data['customer_type']

# month
mark_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

pending_lists = [list() for _ in mark_list]

for i, mark in enumerate(mark_list):
	for thing in test_data['arrival_date_month']:
		if thing == mark:
			pending_lists[i].append(1)
		else:
			pending_lists[i].append(0)

for i, mark in enumerate(mark_list):
	test_data['month_is_{}'.format(mark)] = pending_lists[i]

del mark_list
del pending_lists

print(test_data)

test_data.to_csv('../Data/test_data.csv', index=False)