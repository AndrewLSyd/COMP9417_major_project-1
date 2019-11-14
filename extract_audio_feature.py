import pandas as pd
import glob
import numpy as np
from datetime import datetime


def convert_date_to_week(date_to_convert):
    global start_date
    elasped_days = date_to_convert - start_date
    return int(elasped_days.days / 7) + 1


weekly_silent_ratio = {}
weekly_noisy_ratio = {}
path = 'C:/Users/brs97/PycharmProjects/comp9417/StudentLife_Dataset/Inputs/sensing/audio'  # use your path
all_files = glob.glob(path + "/*.csv")
for file in all_files:
    uid = file.split("_")[-1].rstrip(".csv")
    weekly_silent_ratio[uid] = {}
    weekly_noisy_ratio[uid] = {}
    df = pd.read_csv(file)
    audios = np.array(df)
    weekly_total_detections = {}

    daily_activity = {}
    start_date = datetime.fromtimestamp(audios[0][0]).date()
    for audio in audios:
        date = datetime.fromtimestamp(audio[0]).date()
        week = convert_date_to_week(date)
        audio_level = audio[1]
        if week not in weekly_total_detections:
            weekly_total_detections[week] = 1
        else:
            weekly_total_detections[week] += 1
        if audio_level == 0:
            if week not in weekly_silent_ratio[uid]:
                weekly_silent_ratio[uid][week] = 1
            else:
                weekly_silent_ratio[uid][week] += 1
        elif audio_level == 2:
            if week not in weekly_noisy_ratio[uid]:
                weekly_noisy_ratio[uid][week] = 1
            else:
                weekly_noisy_ratio[uid][week] += 1
    for week in weekly_total_detections:
        if week in weekly_silent_ratio[uid]:
            weekly_silent_ratio[uid][week] /= weekly_total_detections[week]
        if week in weekly_noisy_ratio[uid]:
            weekly_noisy_ratio[uid][week] /= weekly_total_detections[week]

feature1_df = pd.DataFrame.from_dict(weekly_silent_ratio, orient='index')
column1_names = ["audio_silent_ratio_wk_" + str(wk) for wk in range(1, 11)]
feature1_df.columns = column1_names

feature2_df = pd.DataFrame.from_dict(weekly_noisy_ratio, orient='index')
column2_names = ["audio_noisy_ratio_wk_" + str(wk) for wk in range(1, 11)]
feature2_df.columns = column2_names

combined_df = pd.concat([feature1_df, feature2_df], axis=1)
combined_df.to_csv("features_audio.csv")

