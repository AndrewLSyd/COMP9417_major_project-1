import subprocess
# files to set up for user
rscript_location = "C:/Program Files/R/R-3.6.1/bin/Rscript.exe"
file_location    = "S:/Users/Aaron Blackwell/Documents/GitHub/COMP9417_major_project/Code/"

# process combine raw log data into individual data frames
subprocess.call([rscript_location, file_location + "01_import_data.R"] )

# extra features from raw log data
subprocess.call(["python", file_location + "02_feature_engineering_activity.py"])
subprocess.call(["python", file_location + "02_feature_engineering_audio.py"])
subprocess.call(["python", file_location + "02_feature_engineering_bluetooth.py"])
subprocess.call(["python", file_location + "02_feature_engineering_conversation.py"])
subprocess.call([rscript_location, file_location + "02_feature_engineering_gps_wifi.R"] )
subprocess.call(["python", file_location + "02_feature_engineering_phonecharge.py"])
subprocess.call(["python", file_location + "02_feature_engineering_phonelock.py"])
subprocess.call([rscript_location, file_location + "02_feature_engineering_sleep.R"] )

# split data into train and test data sets
subprocess.call(["python", file_location + "03_comb_features_train_test_split.py"])

# fit various models and save results
subprocess.call(["python", file_location + "04_baseline_model.py"])
subprocess.call(["python", file_location + "04_h2o_glm.py"])
subprocess.call(["python", file_location + "04_knn.py"])
subprocess.call(["python", file_location + "04_SVM.py"])
subprocess.call(["python", file_location + "04_xgb_model.py"])
