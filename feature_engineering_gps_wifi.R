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
DATE_TERM_START <- strptime("18.03.2013", format = "%d.%m.%Y")

# options(dplyr.width = Inf)
# head(input_data$gps)
# view(input_data$gps)

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
    replace_na(list(time_elapsed = 1200))

  # cleanup
  # the lagged time value for the first entry for each uid will lead to
  # non-sensical values. assume the time elapsed for the first and last entries is
  # 20m
  # replacing first obs/last obs (for each uid), NANs and negative with 1200s
  df <- create_first_last_obs(df)
  df <- df %>% replace_na(list(time_elapsed = 1200))
  df[df$time_elapsed < 0, ]$time_elapsed = 1200
  df[df$first_obs, ]$time_elapsed = 1200

  df <- select(df, -one_of(c("first_obs", "last_obs")))
  return(df)
}

input_data$gps <- create_time_elapsed(input_data$gps)
input_data$wifi_location <- create_time_elapsed(input_data$wifi_location)

head(input_data$wifi_location)

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

# FEATURES: speed -----------------------------------------------------------
# average speed in each week
speed_mean <- input_data$gps %>%
    group_by(uid, week) %>%
    summarise(speed_mean = mean(speed, na.rm = TRUE))
features_wifi_gps <- spread_join(tibble(), speed_mean, "speed_mean",  TRUE)

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
# distance moved outdoorss using provider == "gps" as a proxy
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

# FEATURES : bearing ---------------------------------------------------------
# bearing_compass <- input_data$gps %>%
#   transmute(bearing_compass = round(bearing / 90)
#
# glimpse(features_wifi_gps)
