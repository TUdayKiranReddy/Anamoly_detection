import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
import pandas as pd

df = pd.read_csv('sensor_data.csv')
df = df.sort_values(by='time_stamp')
#print(df)
id = df['Id'].to_numpy()
dt = [datetime.datetime.strptime(i, '%H:%M') 
				for i in df['time_stamp']]
a = df['sensor_a'].to_numpy()
b = df['sensor_b'].to_numpy()
c = df['sensor_c'].to_numpy()
d = df['sensor_d'].to_numpy()

# file = open('dataset.txt', 'r')
# a =[]
# b = []
# c = []
# d = []
# e = []
# dt = []
# for data in file.readlines():
# 	attr = data[:-1].split(',')
# 	a.append(int(attr[0]))
# 	b.append(float(attr[1]))
# 	c.append(int(attr[2]))
# 	d.append(float(attr[3]))
# 	e.append(float(attr[4]))
# 	dt.append(datetime.datetime.strptime(attr[5]+ ' '+ attr[6], '%Y-%m-%d %H:%M:%S'))

plt.rc('font', size=12)
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 6))

ax[0].stem(dt, a)
ax[0].xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))

ax[1].plot(dt, b)
ax[1].xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))

ax[2].plot(dt, c)
ax[2].xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
plt.show()