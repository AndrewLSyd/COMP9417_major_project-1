import pandas as pd
import os
import datetime
import time
import pytz
import re
import numpy as np


def import_user_data(file):
    with open(file, 'r') as f:
        df = pd.read_csv(file)
    return df

df_list = []
username_list = []
input = 'StudentLife_Dataset/Inputs/sensing/phonelock/'
for root, dirs, files in os.walk(input):
    for file in files:
        df_list.append(import_user_data(input + file))
        username_list.append(re.search('u[0-9]+', file).group())


def week_def():
    data_start = datetime.datetime.strptime("2013-03-25", "%Y-%m-%d")
    data_start = pytz.timezone("Etc/GMT+5").localize(data_start)
    week_start = []
    for i in range(0,10):
        diff = datetime.timedelta(days=(7*i))
        week_start.append(data_start + diff)
    return week_start

week_list = week_def()


def parse_start_duration(df):
    duration_column = []
    start_datetime = []
    end_datetime = []
    for index, row in df.iterrows():
        start = datetime.datetime.fromtimestamp(row['start'], tz=pytz.timezone("Etc/GMT+5"))
        start_datetime.append(start)
        end = datetime.datetime.fromtimestamp(row['end'], tz=pytz.timezone("Etc/GMT+5"))
        end_datetime.append(end)
        duration_column.append(row['end'] - row['start'])
    df['duration'] = duration_column
    df['start_datetime'] = start_datetime
    df['end_datetime'] = end_datetime

    
for df in df_list:
    parse_start_duration(df)


# Mean locktime by week
data = []
for i, user in enumerate(df_list):
    data.append([username_list[i]])
    for j in range(0, 10):
        var = []
        for index, row in user.iterrows():
            if j < 9 and row['start_datetime'] > week_list[j] and row['start_datetime'] < week_list[j+1]:
                var.append(row['duration'])
            elif row['start_datetime'] > week_list[j]:
                var.append(row['duration'])
        try:
            var = sum(var) / len(var)
            data[i].append(var)
        except ZeroDivisionError:
            data[i].append(0)

df_locktime_mean = pd.DataFrame(data, columns = ['uid', 'locktime_mean_wk_1', 'locktime_mean_wk_2', 'locktime_mean_wk_3',
                                                'locktime_mean_wk_4', 'locktime_mean_wk_5', 'locktime_mean_wk_6',
                                                'locktime_mean_wk_7', 'locktime_mean_wk_8', 'locktime_mean_wk_9',
                                                'locktime_mean_wk_10'])
df_locktime_mean.to_csv('StudentLife_Dataset/out/locktime_mean.csv', index=False)


# Count phone lock by week
data = []
for i, user in enumerate(df_list):
    data.append([username_list[i]])
    for j in range(0, 10):
        var = 0
        for index, row in user.iterrows():
            if j < 9 and row['start_datetime'] > week_list[j] and row['start_datetime'] < week_list[j+1]:
                var += 1
            elif row['start_datetime'] > week_list[j]:
                var += 1
        data[i].append(var)

df_locktime_count = pd.DataFrame(data, columns=['uid', 'locktime_count_wk_1', 'locktime_count_wk_2', 'locktime_count_wk_3',
                                               'locktime_count_wk_4', 'locktime_count_wk_5', 'locktime_count_wk_6', 
                                               'locktime_count_wk_7', 'locktime_count_wk_8', 'locktime_count_wk_9', 
                                               'locktime_count_wk_10'])
df_locktime_count.to_csv('StudentLife_Dataset/out/locktime_count.csv', index=False)


# Median locktime by week
data = []
for i, user in enumerate(df_list):
    data.append([username_list[i]])
    for j in range(0, 10):
        var = []
        for index, row in user.iterrows():
            if j < 9 and row['start_datetime'] > week_list[j] and row['start_datetime'] < week_list[j+1]:
                var.append(row['duration'])
            elif row['start_datetime'] > week_list[j]:
                var.append(row['duration'])
        try:
            data[i].append(np.percentile(var, 50))
        except:
            data[i].append(0)

df_locktime_median = pd.DataFrame(data, columns = ['uid', 'locktime_median_wk_1', 'locktime_median_wk_2', 'locktime_median_wk_3',
                                                'locktime_median_wk_4', 'locktime_median_wk_5', 'locktime_median_wk_6',
                                                'locktime_median_wk_7', 'locktime_median_wk_8', 'locktime_median_wk_9',
                                                'locktime_median_wk_10'])
df_locktime_median.to_csv('StudentLife_Dataset/out/locktime_median.csv', index=False)


# First quartile locktime by week
data = []
for i, user in enumerate(df_list):
    data.append([username_list[i]])
    for j in range(0, 10):
        var = []
        for index, row in user.iterrows():
            if j < 9 and row['start_datetime'] > week_list[j] and row['start_datetime'] < week_list[j+1]:
                var.append(row['duration'])
            elif row['start_datetime'] > week_list[j]:
                var.append(row['duration'])
        try:
            data[i].append(np.percentile(var, 25))
        except:
            data[i].append(0)

df_locktime_q1 = pd.DataFrame(data, columns = ['uid', 'locktime_q1_wk_1', 'locktime_q1_wk_2', 'locktime_q1_wk_3',
                                                'locktime_q1_wk_4', 'locktime_q1_wk_5', 'locktime_q1_wk_6',
                                                'locktime_q1_wk_7', 'locktime_q1_wk_8', 'locktime_q1_wk_9',
                                                'locktime_q1_wk_10'])
df_locktime_q1.to_csv('StudentLife_Dataset/out/locktime_q1.csv', index=False)


# Third quartile locktime by week
data = []
for i, user in enumerate(df_list):
    data.append([username_list[i]])
    for j in range(0, 10):
        var = []
        for index, row in user.iterrows():
            if j < 9 and row['start_datetime'] > week_list[j] and row['start_datetime'] < week_list[j+1]:
                var.append(row['duration'])
            elif row['start_datetime'] > week_list[j]:
                var.append(row['duration'])
        try:
            data[i].append(np.percentile(var, 75))
        except:
            data[i].append(0)

df_locktime_q3 = pd.DataFrame(data, columns = ['uid', 'locktime_q3_wk_1', 'locktime_q3_wk_2', 'locktime_q3_wk_3',
                                                'locktime_q3_wk_4', 'locktime_q3_wk_5', 'locktime_q3_wk_6',
                                                'locktime_q3_wk_7', 'locktime_q3_wk_8', 'locktime_q3_wk_9',
                                                'locktime_q3_wk_10'])
df_locktime_q3.to_csv('StudentLife_Dataset/out/locktime_q3.csv', index=False)


# Min locktime by week
data = []
for i, user in enumerate(df_list):
    data.append([username_list[i]])
    for j in range(0, 10):
        var = 86400
        for index, row in user.iterrows():
            if j < 9 and row['start_datetime'] > week_list[j] and row['start_datetime'] < week_list[j+1]:
                if row['duration'] < var:
                    var = row['duration']
            elif row['start_datetime'] > week_list[j]:
                if row['duration'] < var:
                    var = row['duration']
        if var == 86400:
            var = 0
        data[i].append(var)

df_locktime_min = pd.DataFrame(data, columns=['uid', 'locktime_min_wk_1', 'locktime_min_wk_2', 'locktime_min_wk_3',
                                               'locktime_min_wk_4', 'locktime_min_wk_5', 'locktime_min_wk_6', 
                                               'locktime_min_wk_7', 'locktime_min_wk_8', 'locktime_min_wk_9', 
                                               'locktime_min_wk_10'])
df_locktime_min.to_csv('StudentLife_Dataset/out/locktime_min.csv', index=False)


# Max locktime by week
data = []
for i, user in enumerate(df_list):
    data.append([username_list[i]])
    for j in range(0, 10):
        var = 0
        for index, row in user.iterrows():
            if j < 9 and row['start_datetime'] > week_list[j] and row['start_datetime'] < week_list[j+1]:
                if row['duration'] > var:
                    var = row['duration']
            elif row['start_datetime'] > week_list[j]:
                if row['duration'] > var:
                    var = row['duration']
        data[i].append(var)

df_locktime_max = pd.DataFrame(data, columns=['uid', 'locktime_max_wk_1', 'locktime_max_wk_2', 'locktime_max_wk_3',
                                               'locktime_max_wk_4', 'locktime_max_wk_5', 'locktime_max_wk_6', 
                                               'locktime_max_wk_7', 'locktime_max_wk_8', 'locktime_max_wk_9', 
                                               'locktime_max_wk_10'])
df_locktime_max.to_csv('StudentLife_Dataset/out/locktime_max.csv', index=False)
