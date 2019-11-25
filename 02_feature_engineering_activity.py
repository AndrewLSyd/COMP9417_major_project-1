import pandas as pd
import glob
import numpy as np
from datetime import datetime
from pytz import timezone


def convert_date_to_week(date_to_convert):
    global start_date
    elasped_days = date_to_convert - start_date
    return int(elasped_days.days / 7) + 1

# initialise variables
weekly_stationary_ratio = {}
weekly_running_ratio = {}
tz = timezone('Etc/GMT+5')
start_date = tz.localize(datetime(2013, 3, 25)).date()
path = 'C:/Users/brs97/PycharmProjects/comp9417/StudentLife_Dataset/Inputs/sensing/activity'  # use your path

all_files = glob.glob(path + "/*.csv")
for file in all_files:
    # get student activity
    uid = file.split("_")[-1].rstrip(".csv")
    weekly_stationary_ratio[uid] = {}
    weekly_running_ratio[uid] = {}
    df = pd.read_csv(file)
    activities = np.array(df)
    weekly_total_detections = {}
    
    # split activity into count, stationary and running activity
    for activity in activities:
        date = datetime.fromtimestamp(activity[0]).date()
        week = convert_date_to_week(date)
        activity_level = activity[1]
        if week not in weekly_total_detections:
            weekly_total_detections[week] = 0
            weekly_stationary_ratio[uid][week] = 0
            weekly_running_ratio[uid][week] = 0
        weekly_total_detections[week] += 1
        if activity_level == 0:
            weekly_stationary_ratio[uid][week] += 1
        elif activity_level == 2:
            weekly_running_ratio[uid][week] += 1
    # summarise at a weekly level
    for week in weekly_total_detections:
        if week in weekly_stationary_ratio[uid]:
            weekly_stationary_ratio[uid][week] /= weekly_total_detections[week]
        if week in weekly_running_ratio[uid]:
            weekly_running_ratio[uid][week] /= weekly_total_detections[week]

# write out to csv
feature1_df = pd.DataFrame.from_dict(weekly_stationary_ratio, orient='index')
feature1_df = feature1_df.reindex(sorted(feature1_df.columns), axis=1)
column1_names = ["activity_stationary_ratio_wk_" + str(wk) for wk in range(1, 11)]
feature1_df.columns = column1_names

feature2_df = pd.DataFrame.from_dict(weekly_running_ratio, orient='index')
feature2_df = feature2_df.reindex(sorted(feature2_df.columns), axis=1)
column2_names = ["activity_running_ratio_wk_" + str(wk) for wk in range(1, 11)]
feature2_df.columns = column2_names

combined_df = pd.concat([feature1_df, feature2_df], axis=1)
combined_df.to_csv("features_activity.csv")

