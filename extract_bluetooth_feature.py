import pandas as pd
import glob
import numpy as np
from datetime import datetime


def convert_date_to_week(date_to_convert):
    global start_date
    elasped_days = date_to_convert - start_date
    return int(elasped_days.days / 7) + 1


weekly_avg_bluetooth = {}
path = 'C:/Users/brs97/PycharmProjects/comp9417/StudentLife_Dataset/Inputs/sensing/bluetooth'  # use your path
all_files = glob.glob(path + "/*.csv")
for file in all_files:
    uid = file.split("_")[-1].rstrip(".csv")
    weekly_avg_bluetooth[uid] = {}
    df = pd.read_csv(file)
    bluetooths = np.array(df)
    daily_bluetooth = {}
    start_date = datetime.fromtimestamp(bluetooths[0][0]).date()
    for bluetooth in bluetooths:
        date = datetime.fromtimestamp(bluetooth[0]).date()
        if date not in daily_bluetooth:
            daily_bluetooth[date] = []
        if bluetooth[1] not in daily_bluetooth[date]:
            daily_bluetooth[date].append(bluetooth[1])
    for date in daily_bluetooth:
        week = convert_date_to_week(date)
        if week not in weekly_avg_bluetooth:
            weekly_avg_bluetooth[uid][week] = len(daily_bluetooth[date]) / 7
        else:
            weekly_avg_bluetooth[uid][week] += len(daily_bluetooth[date]) / 7

feature_df = pd.DataFrame.from_dict(weekly_avg_bluetooth, orient='index')
column_names = ["bluetooth_avg_wk_" + str(wk) for wk in range(1, 11)]
feature_df.columns = column_names

feature_df.to_csv("features_bluetooth.csv")
