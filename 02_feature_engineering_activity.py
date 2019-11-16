import pandas as pd
import glob
import numpy as np
from datetime import datetime


def convert_date_to_week(date_to_convert):
    global start_date
    elasped_days = date_to_convert - start_date
    return int(elasped_days.days / 7) + 1


weekly_stationary_ratio = {}
weekly_running_ratio = {}
path = 'C:/Users/brs97/PycharmProjects/comp9417/StudentLife_Dataset/Inputs/sensing/activity'  # use your path
all_files = glob.glob(path + "/*.csv")
for file in all_files:
    uid = file.split("_")[-1].rstrip(".csv")
    weekly_stationary_ratio[uid] = {}
    weekly_running_ratio[uid] = {}
    df = pd.read_csv(file)
    activities = np.array(df)
    weekly_total_detections = {}

    daily_activity = {}
    start_date = datetime.fromtimestamp(activities[0][0]).date()
    for activity in activities:
        date = datetime.fromtimestamp(activity[0]).date()
        week = convert_date_to_week(date)
        activity_level = activity[1]
        if week not in weekly_total_detections:
            weekly_total_detections[week] = 1
        else:
            weekly_total_detections[week] += 1
        if activity_level == 0:
            if week not in weekly_stationary_ratio[uid]:
                weekly_stationary_ratio[uid][week] = 1
            else:
                weekly_stationary_ratio[uid][week] += 1
        elif activity_level == 2:
            if week not in weekly_running_ratio[uid]:
                weekly_running_ratio[uid][week] = 1
            else:
                weekly_running_ratio[uid][week] += 1
    for week in weekly_total_detections:
        if week in weekly_stationary_ratio[uid]:
            weekly_stationary_ratio[uid][week] /= weekly_total_detections[week]
        if week in weekly_running_ratio[uid]:
            weekly_running_ratio[uid][week] /= weekly_total_detections[week]

feature1_df = pd.DataFrame.from_dict(weekly_stationary_ratio, orient='index')
column1_names = ["activity_stationary_ratio_wk_" + str(wk) for wk in range(1, 11)]
feature1_df.columns = column1_names

feature2_df = pd.DataFrame.from_dict(weekly_running_ratio, orient='index')
column2_names = ["activity_running_ratio_wk_" + str(wk) for wk in range(1, 11)]
feature2_df.columns = column2_names

combined_df = pd.concat([feature1_df, feature2_df], axis=1)
combined_df.to_csv("features_activity.csv")

