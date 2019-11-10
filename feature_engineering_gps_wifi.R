# COMP9417 Group Assignment
# Andrew Lau
# Feature engineering for wifi, wifi location and GPS input datasets

DATE_TERM_START = as.Date("17/3/2013", format="%d/%m/%y")

# speed_mean - mean of speed observations in each week
# speed_sd - standard deviation of speed observations in each week
# speed_max - max of speed observations in each week
# travelstate_time_stationary - seconds spend stationary in each week
# travelstate_time_moving - seconds spend moving in each week
# outdoor_time - seconds spent outdoors (using GPS as a proxy, as per the paper)
# indoor_time - seconds spent indoors (using WiFi as a proxy, as per the paper)
# distance_moved_indoors
# distance_moved_outdoors
# bearing - grouped into N E S W for outdoor data only
# number of locations visited
# if time, consider grouping the locations (e.g. gym, libaray)

# time spent in various travel states each week
    # break down by

# number of locations visited in each week
# indoor and outdoor mobility
    # distance travelled

# input_data_backup = input_data
head(input_data$gps)
view(input_data$gps)

########################################
# HELPER FUNCTIONS
########################################
spread_join <- function(df, var, var_str, first=FALSE){
    # spreads out the tibble, var, joins it to the df datafram
    var <- var %>%
        spread(week, var_str) %>%
        rename_at(vars(-uid), funs(paste0(var_str, "_wk_", .)))
    
    if (first){
        return(as_tibble(var))
    }
    
    df <- left_join(df, var, by=c("uid"))
    return(df)
}


########################################
# PREPARING DATA
########################################
# create field for start of the week
# date of monday of the start of the week
input_data$gps$week = cut(as.Date(dates), "week")
# convert to week number in the term
input_data$gps$week = round(difftime(input_data$gps$week, strptime("18.03.2013", format = "%d.%m.%Y"), units="weeks"))

# creating time elapsed field (seconds):
    # each measurement is a snapshot in time. we need a way to summarise by
    # amount of time. we do this by assuming the user is at the location/doing
    # the activity from the halfway between the previous time and the next time.
# time_elapsed is calculated as:
    # 0.5 * (time - time_lag) + 0.5 * (time_lead - time)
# 1. sort by uid then time
input_data$gps <- input_data$gps %>% 
    arrange(uid, time) %>%
    # 2. create lagged/led time fields
    mutate(time_lag = lag(time)) %>%
    mutate(time_lead = lead(time)) %>%
    # 3. create time elapsed variable 
    mutate(time_elapsed = 0.5 * (time - time_lag) + 0.5 * (time_lead - time))

# 4. cleanup
# the lagged time value for the first entry for each uid will lead to
# non-sensical values. assume the time elapsed for the first and last entries is
# 20m
input_data$gps <- input_data$gps %>% replace_na(list(time_elapsed = 1200))
input_data$gps[input_data$gps$time_elapsed < 0, ]$time_elapsed = 1200

# dropping uneeded columns
input_data$gps <- input_data$gps %>% select(-one_of(c("time_lag", "time_lead")))

########################################
# FEATURES - SPEED
########################################
# average speed in each week
speed_mean <- input_data$gps %>%
    group_by(uid, week) %>%
    summarise(speed_mean = mean(speed))
features_wifi_gps <- spread_join(tibble(), speed_mean, "speed_mean",  TRUE)

# max speed in each week
speed_max <- input_data$gps %>%
    group_by(uid, week) %>% 
    summarise(speed_max = max(speed))
features_wifi_gps <- spread_join(features_wifi_gps, speed_max, "speed_max")

# standard deviation of speed in each week
speed_sd <- input_data$gps %>%
    group_by(uid, week) %>% 
    summarise(speed_sd = sd(speed))
features_wifi_gps <- spread_join(features_wifi_gps, speed_sd, "speed_sd")

########################################
# FEATURES - TRAVELSTATE
########################################
# total time spent stationary every week
travelstate_time_stationary <- input_data$gps %>%
    filter(travelstate == "stationary") %>%
    group_by(uid, week) %>%
    summarise(travelstate_time_stationary = sum(time_elapsed))
features_wifi_gps <- spread_join(features_wifi_gps, travelstate_time_stationary, "travelstate_time_stationary")

# total time spent moving every week
travelstate_time_moving <- input_data$gps %>%
    filter(travelstate == "moving") %>%
    group_by(uid, week) %>%
    summarise(travelstate_time_moving = sum(time_elapsed))
features_wifi_gps <- spread_join(features_wifi_gps, travelstate_time_moving, "travelstate_time_moving")
