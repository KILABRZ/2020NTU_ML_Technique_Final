import pandas as pd
import numpy as np

train_data = pd.read_csv('../RawData/train.csv')
train_label = pd.read_csv('../RawData/train_label.csv')

train_data['arrival_date_month'] = train_data['arrival_date_month'].replace({
	'January' : 1, 'February' : 2, 'March' : 3, 'April' : 4, 'May' : 5,
	'June' : 6, 'July' : 7, 'August' : 8, 'September' : 9, 'October' : 10,
	'November' : 11, 'December' : 12
})
train_data = train_data.sort_values(by=['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'])
train_data = train_data[train_data['is_canceled'] == 0]
N = len(train_data)

adr = train_data['adr'].to_numpy().reshape(N, 1)
stay_days = train_data['stays_in_weekend_nights'].to_numpy() + train_data['stays_in_week_nights'].to_numpy()
stay_days = stay_days.reshape(N, 1)
start_d = train_data['arrival_date_day_of_month'].to_numpy().reshape(N, 1)
start_m = train_data['arrival_date_month'].to_numpy().reshape(N, 1)
start_y = train_data['arrival_date_year'].to_numpy().reshape(N, 1)

start_timestamp = np.concatenate((start_y, start_m, start_d), axis=1)
revenue = adr * stay_days

compacted_data = np.concatenate((start_timestamp , revenue), axis=1)

days, spliter = np.unique(start_timestamp, axis=0, return_index=True)

day_record = np.split(revenue, spliter)[1:]

revenue_list = list()

for record in day_record:
	revenue_list.append(record.sum())

revenue_list = np.array(revenue_list)
label = train_label['label'].to_numpy()

import matplotlib.pyplot as plt

plt.scatter(label, revenue_list)

plt.show()