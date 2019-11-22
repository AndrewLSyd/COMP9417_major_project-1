# Data Dictionary


## Wifi/GPS
### Speed based features:
Speed of movement in a week may be indicative of levels of exercise/psychological
* speed_mean - mean of speed observations in each week  

* speed_sd - standard deviation of speed observations in each week  
* speed_max - max of speed observations in each week  

### travelstate based features:
* travelstate_time_stationary - seconds spend stationary in each week  
* travelstate_time_moving - seconds spend moving in each week

### Phonelock
* locktime_count - number of times phone has been locked each week
* locktime_max - longest continuous time locked each week
* locktime_min - shortest continuous time locked each week
* locktime_mean - average time locked each week
* locktime_median - median time locked each week
* locktime_q1 - first quartile phone locked time each week
* locktime_q3 - third quartile phone locked time each week

### Phonecharge
* chargetime_count - number of times phone has been plugged in to charge each week
* chargetime_max - longest continuous time charging has occurred each week
* chargetime_min - shortest continuous time charging has occurred each week
* chargetime_mean - average time phone has been charging each week
* chargetime_median - median time phone has been charging each week
* chargetime_q1 - first quartile phone charge time each week
* chargetime_q3 - third quartile phone charge time each week

### Bluetooth
* bluetooth_avg - average number of unique bluetooth devices (daily) each week

### Activity
* activity_stationary_ratio - proportion of the user's stationary time in each week
* activity_running_ratio - proportion of the user's running time in each week

### Conversation
* conversation_hours - total hours of conversation that the user participate in each week
* conversation_freq - total number of conversation that the user participate in each week
