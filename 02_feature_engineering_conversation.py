import pandas as pd
import glob
import numpy as np
from datetime import datetime
from pytz import timezone


def convert_date_to_week(date_to_convert):
    global start_date
    elasped_days = date_to_convert - start_date
    return int(elasped_days.days / 7) + 1


weekly_conversation_duration = {}
weekly_conversation_freq = {}
tz = timezone('Etc/GMT+5')
start_date = tz.localize(datetime(2013, 3, 25)).date()
path = 'C:/Users/brs97/PycharmProjects/comp9417/StudentLife_Dataset/Inputs/sensing/conversation'  # use your path
all_files = glob.glob(path + "/*.csv")
for file in all_files:
    uid = file.split("_")[-1].rstrip(".csv")
    weekly_conversation_duration[uid] = {}
    weekly_conversation_freq[uid] = {}
    df = pd.read_csv(file)
    conversations = np.array(df)
    for conversation in conversations:
        start = datetime.fromtimestamp(conversation[0])
        end = datetime.fromtimestamp(conversation[1])
        duration = end - start
        week = convert_date_to_week(start.date())
        if week not in weekly_conversation_duration[uid]:
            weekly_conversation_duration[uid][week] = 0
            weekly_conversation_freq[uid][week] = 0
        weekly_conversation_duration[uid][week] += duration.seconds / 3600
        weekly_conversation_freq[uid][week] += 1

feature1_df = pd.DataFrame.from_dict(weekly_conversation_duration, orient='index')
feature1_df = feature1_df.reindex(sorted(feature1_df.columns), axis=1)
feature1_column_names = ["conversation_hours_wk_" + str(wk) for wk in range(1, 11)]
feature1_df.columns = feature1_column_names

feature2_df = pd.DataFrame.from_dict(weekly_conversation_freq, orient='index')
feature2_df = feature2_df.reindex(sorted(feature2_df.columns), axis=1)
feature2_column_names = ["conversation_freq_wk_" + str(wk) for wk in range(1, 11)]
feature2_df.columns = feature2_column_names

combined_df = pd.concat([feature1_df, feature2_df], axis=1)
combined_df.to_csv("features_conversation.csv")
