import pandas as pd
import glob
import numpy as np
from datetime import datetime
from pytz import timezone


def convert_date_to_week(date_to_convert):
    global start_date
    elasped_days = date_to_convert - start_date
    return int(elasped_days.days / 7) + 1


weekly_avg_bluetooth = {}
tz = timezone('Etc/GMT+5')
start_date = tz.localize(datetime(2013, 3, 25)).date()
path = 'C:/Users/brs97/PycharmProjects/comp9417/StudentLife_Dataset/Inputs/sensing/bluetooth'  # use your path
all_files = glob.glob(path + "/*.csv")
for file in all_files:
    uid = file.split("_")[-1].rstrip(".csv")
    weekly_avg_bluetooth[uid] = {}
    df = pd.read_csv(file)
    bluetooths = np.array(df)
    daily_bluetooth = {}
    for bluetooth in bluetooths:
        date = datetime.fromtimestamp(bluetooth[0], tz).date()
        if date not in daily_bluetooth:
            daily_bluetooth[date] = []
        if bluetooth[1] not in daily_bluetooth[date]:
            daily_bluetooth[date].append(bluetooth[1])
    for date in daily_bluetooth:
        week = convert_date_to_week(date)
        if week not in weekly_avg_bluetooth[uid]:
            weekly_avg_bluetooth[uid][week] = 0
        weekly_avg_bluetooth[uid][week] += len(daily_bluetooth[date]) / 7

feature_df = pd.DataFrame.from_dict(weekly_avg_bluetooth, orient='index')
feature_df = feature_df.reindex(sorted(feature_df.columns), axis=1)
column_names = ["bluetooth_avg_wk_" + str(wk) for wk in range(1, 11)]
feature_df.columns = column_names

feature_df.to_csv("features_bluetooth.csv")
