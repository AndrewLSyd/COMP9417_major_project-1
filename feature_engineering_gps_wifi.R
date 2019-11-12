# COMP9417 Group Assignment
# AUTHOR: Andrew Lau
# PURPOSE: Feature engineering for wifi, wifi location and GPS input datasets

# SUMMARY OF FEATURES ----------------------------------------------------------
# speed based features:
# speed of movement in a week may be indicate of levels of exercise/psych.
# health
# speed_mean - mean of speed observations in each week
# speed_sd - standard deviation of speed observations in each week
# speed_max - max of speed observations in each week

# travelstate based features:
# speed of movement in a week may be indicate of levels of exercise/psych.
# health
# travelstate_time_stationary - seconds spend stationary in each week
# travelstate_time_moving - seconds spend moving in each week

# indoor/outdoor based features:
# the paper uses network == 'gps' as a proxy for being outdoors. amount of time
# spent indoors/outdoors could be linked to psych. health
# outdoor_time - seconds spent outdoors (using GPS as a proxy, as per the paper)
  # in each week
# indoor_time - seconds spent indoors (using WiFi as a proxy, as per the paper)
  # in each week
# distance_moved_indoors - distance moved indoors (using WiFi as a proxy, as per
  # the paper) in each week
# distance_moved_outdoors - distance moved outdoors (using WiFi as a proxy, as
  # per the paper) in each week

# altitude based features:
# altitude_mean - mean altitude in each week
# altitude_sd - standard deviation of altitude in each week
# altitude_min - minimum of altitude in each week
# altitude_max - maximum of altitude in each week

# locations based features
# location_count - number of diffrent locations visited in each week
# if we have time, we can consider grouping the locations (e.g. gym, library)
# ^ this will be very time consuming though

# bearing - grouped into N E S W for outdoor data only

# CONSTANTS --------------------------------------------------------------------
FPATH_OUT <- "preprocessed_data/features_gps_wifi.csv"
DATE_TERM_START <- strptime("18.03.2013", format = "%d.%m.%Y")
# value used when time_elapsed cannot be calculated. 20m represents the median
# value of time elapsed
TIME_ELAPSED_IMPUTE <- 1200
TIME_ELAPSED_CAP <- 1800
DIST_MOVED_CAP <- 1.045034e-02

options(dplyr.width = Inf)
# head(input_data$gps)
# view(input_data$gps)
# input_data_backup = input_data


# HELPER FUNCTIONS -------------------------------------------------------------
spread_join <- function(df, var, var_str, first=FALSE){
    # spreads out the tibble, var, and joins it to the tibble df
    # spread out the tibble so that there is a row for each uid and each summary
    # has a new column for each week
    var <- var %>%
        spread(week, var_str) %>%
        rename_at(vars(-uid), funs(paste0(var_str, "_wk_", .)))
    if (first){
        return(as_tibble(var))
    }
    df <- left_join(df, var, by=c("uid"))
    return(df)
}

# PREPARING DATA: week number --------------------------------------------------
create_week_num <- function(df){
  # create field for start of the week - date of monday of the start of the week
  # convert to week number in the term
  df <- df %>%
    mutate(date=as.Date(time)) %>%
    mutate(
      week = floor(
        difftime(
          time1 = date,
          time2 = DATE_TERM_START,
          units = "weeks"))
    ) %>%
    select(-one_of(c("date")))
  return(df)
}

input_data$gps <- create_week_num(input_data$gps)
input_data$wifi_location <- create_week_num(input_data$wifi_location)

# PREPARING DATA: time elapsed -------------------------------------------------
create_first_last_obs <- function(df){
  # creating field for first and last obs
  df <- df %>%
    mutate(first_obs = lag(uid) != uid) %>%
    mutate(last_obs = lead(uid) != uid) %>%
    replace_na(list(first_obs = TRUE, last_obs = TRUE))
  return(df)
}

create_time_elapsed <- function(df){
  # creating time elapsed field (seconds):
  # time_elapsed is calculated as: time - previous time
  df <- df %>%
    arrange(uid, time) %>%  # sort by uid then time
    mutate(time_elapsed = time - lag(time)) %>%
    replace_na(list(time_elapsed = TIME_ELAPSED_IMPUTE))

  # cleanup
  # the lagged time value for the first entry for each uid will lead to
  # non-sensical values. assume the time elapsed for the first and last entries
  # is 20m
  # replacing first obs/last obs (for each uid), NANs and negative with 1200s
  df <- create_first_last_obs(df)
  df <- df %>% replace_na(list(time_elapsed = TIME_ELAPSED_IMPUTE))
  df[df$time_elapsed < 0, ]$time_elapsed = TIME_ELAPSED_IMPUTE
  df[df$first_obs, ]$time_elapsed = TIME_ELAPSED_IMPUTE

  # there are non-sensically high values for time elapsed, finding something
  # reasonable to cap it at. Have gone with 30m which is ~98th percentile
  df[df$time_elapsed > TIME_ELAPSED_CAP, ]$time_elapsed = TIME_ELAPSED_CAP
  df <- select(df, -one_of(c("first_obs", "last_obs")))
  return(df)
}


input_data$gps <- create_time_elapsed(input_data$gps)
input_data$wifi_location <- create_time_elapsed(input_data$wifi_location)

# quantile(input_data$gps$time_elapsed, c(.50, .75, .9, 0.95, 0.975, .98, .99,
# .995, 1))

# PREPARING DATA: distance moved -----------------------------------------------
input_data$gps <- create_first_last_obs(input_data$gps)
input_data$gps <- input_data$gps %>%
  mutate(dist_moved_long = longitude - lag(longitude)) %>%
  mutate(dist_moved_lat = latitude - lag(latitude)) %>%
  mutate(dist_moved = sqrt(dist_moved_long ** 2 + dist_moved_lat ** 2)) %>%
  # if it's the first obs, zero it out as the previous records is from another
  # uid
  mutate(dist_moved = dist_moved * 0 ** first_obs) %>%
  replace_na(list(dist_moved = 0)) %>%
  select(-one_of(c("first_obs", "last_obs", "dist_moved_lat",
                   "dist_moved_long")))
  # capping dist_moved at the value of the 98th percentile to remove
  # non-sensical values
  # quantile(input_data$gps$dist_moved, c(.50, .75, .9, 0.95, 0.975, .98, .99, .995, 1))

input_data$gps[input_data$gps$dist_moved > DIST_MOVED_CAP, ]$dist_moved = DIST_MOVED_CAP

# FEATURES: speed -----------------------------------------------------------
# average speed in each week
speed_mean <- input_data$gps %>%
    group_by(uid, week) %>%
    summarise(speed_mean = mean(speed, na.rm = TRUE))
features_wifi_gps <- spread_join(tibble(), speed_mean, "speed_mean",  TRUE)
# median speed in each week is of no value as the median value is 0 for most uid

# max speed in each week
speed_max <- input_data$gps %>%
    group_by(uid, week) %>%
    summarise(speed_max = max(speed))
features_wifi_gps <- spread_join(features_wifi_gps, speed_max, "speed_max")

# standard deviation of speed in each week
speed_sd <- input_data$gps %>%
    group_by(uid, week) %>%
    summarise(speed_sd = sd(speed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, speed_sd, "speed_sd")

# FEATURES: travelstate --------------------------------------------------------
# total time spent stationary every week
travelstate_time_stationary <- input_data$gps %>%
    filter(travelstate == "stationary") %>%
    group_by(uid, week) %>%
    summarise(travelstate_time_stationary = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, travelstate_time_stationary,
                                 "travelstate_time_stationary")

# total time spent moving every week
travelstate_time_moving <- input_data$gps %>%
    filter(travelstate == "moving") %>%
    group_by(uid, week) %>%
    summarise(travelstate_time_moving = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, travelstate_time_moving,
                                 "travelstate_time_moving")

# FEATURES: indoor/outdoor -----------------------------------------------------
# the paper uses gps to proxy time spent outdoors
outdoor_time <- input_data$gps %>%
  group_by(uid, week) %>%
  filter(provider == "gps") %>%
  summarise(outdoor_time = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, outdoor_time,
                                 "outdoor_time")
# time spent indoors using provider == "gps" as a proxy
indoor_time <- input_data$gps %>%
  group_by(uid, week) %>%
  filter(provider != "gps") %>%
  summarise(indoor_time = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, indoor_time,
                                 "indoor_time")
# distance moved indoors using provider == "gps" as a proxy
indoor_dist <- input_data$gps %>%
  group_by(uid, week) %>%
  filter(provider != "gps") %>%
  summarise(indoor_dist = sum(dist_moved, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, indoor_dist,
                                 "indoor_dist")
# distance moved outdoors using provider == "gps" as a proxy
outdoors_dist <- input_data$gps %>%
  group_by(uid, week) %>%
  filter(provider != "gps") %>%
  summarise(outdoors_dist = sum(dist_moved, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, outdoors_dist,
                                 "outdoors_dist")

# FEATURES : altitude ----------------------------------------------------------
# mean altitude in each week
altitude_mean <- input_data$gps %>%
  group_by(uid, week) %>%
  summarise(altitude_mean = mean(altitude, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, altitude_mean,
                                 "altitude_mean")
# standard deviation of altitude in each week
altitude_sd <- input_data$gps %>%
  group_by(uid, week) %>%
  summarise(altitude_sd = sd(altitude, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, altitude_sd,
                                 "altitude_sd")

# max of altitude in each week
altitude_max <- input_data$gps %>%
  group_by(uid, week) %>%
  summarise(altitude_max = max(altitude, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, altitude_max,
                                 "altitude_max")

# min of altitude in each week
altitude_min <- input_data$gps %>%
  group_by(uid, week) %>%
  summarise(altitude_min = min(altitude, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, altitude_min,
                                 "altitude_min")

# FEATURES : locations ---------------------------------------------------------
# number of different locations visited each week
location_count <- input_data$wifi_location %>%
  group_by(uid, week, location) %>%
  tally() %>%
  group_by(uid, week) %>%
  tally(name = "location_count")
features_wifi_gps <- spread_join(features_wifi_gps, location_count,
                                 "location_count")

# amount of time spent in the 5 most common locations
# top 10 most common locations are:
# input_data$wifi_location %>%
#   group_by(location) %>%
#   tally() %>%
#   arrange(desc(n))

# in[north-main]	284288
# in[sudikoff]	267098
# in[mclaughlin]	143585
# in[north-park]	118475
# in[massrow]	106341

location_1_time <- input_data$wifi_location %>%
  filter(location == "in[north-main]") %>%
  group_by(uid, week) %>%
  summarise(location_1_time = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, location_1_time,
                                 "location_1_time")

location_2_time <- input_data$wifi_location %>%
  filter(location == "in[sudikoff]") %>%
  group_by(uid, week) %>%
  summarise(location_2_time = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, location_2_time,
                                 "location_2_time")

location_3_time <- input_data$wifi_location %>%
  filter(location == "in[mclaughlin]") %>%
  group_by(uid, week) %>%
  summarise(location_3_time = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, location_3_time,
                                 "location_3_time")

location_4_time <- input_data$wifi_location %>%
  filter(location == "in[north-park]") %>%
  group_by(uid, week) %>%
  summarise(location_4_time = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, location_4_time,
                                 "location_4_time")

location_5_time <- input_data$wifi_location %>%
  filter(location == "in[massrow]") %>%
  group_by(uid, week) %>%
  summarise(location_5_time = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, location_5_time,
                                 "location_5_time")

# FEATURES : bearing -----------------------------------------------------------
# only valid if network is GPS
# group into N E S W
bearing <- input_data$gps %>%
  mutate(bearing = round(bearing / 90) %% 4) %>%
  filter(provider == "gps")

bearing_north_time <- bearing %>%
  filter(bearing == 0) %>%
  group_by(uid, week) %>%
  summarise(bearing_north_time = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, bearing_north_time,
                                 "bearing_north_time")

bearing_east_time <- bearing %>%
  filter(bearing == 1) %>%
  group_by(uid, week) %>%
  summarise(bearing_east_time = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, bearing_east_time,
                                 "bearing_east_time")

bearing_south_time <- bearing %>%
  filter(bearing == 2) %>%
  group_by(uid, week) %>%
  summarise(bearing_south_time = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, bearing_south_time,
                                 "bearing_south_time")

bearing_west_time <- bearing %>%
  filter(bearing == 3) %>%
  group_by(uid, week) %>%
  summarise(bearing_west_time = sum(time_elapsed, na.rm = TRUE))
features_wifi_gps <- spread_join(features_wifi_gps, bearing_west_time,
                                 "bearing_west_time")

# FEATURES : export ------------------------------------------------------------
write_csv(features_wifi_gps, FPATH_OUT)
