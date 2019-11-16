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
# CONSTANTS ---------------------------------------------------------------------
feature_folder <- "preprocessed_data"
feature_name <- "features_gps_wifi.csv"

# this agrees to feature_engineering_sleep.R code :)
term_start <- ymd("20130318", tz = "US/Eastern")
# value used when time_elapsed cannot be calculated. 20m represents the median
# value of time elapsed
time_elapsed_impute <- as.difftime(20, units = "mins")
time_elapsed_cap <- as.difftime(30, units = "mins")
distance_moved_cap <- 0.01045034


# HELPER FUNCTIONS -------------------------------------------------------------
spread_join <- function(df, var, var_str, first=FALSE) {
  # spreads out the tibble, var, and joins it to the tibble df
  # spread out the tibble so that there is a row for each uid and each summary
  # has a new column for each week
  var <-
    var %>%
    spread(week, var_str) %>%
    rename_at(vars(-uid), list(~ paste0(var_str, "_wk_", .)))

  if (first) {
    as_tibble(var)
  } else {
    left_join(df, var, by = c("uid"))
  }
}
# PREPARING DATA: week number --------------------------------------------------
create_week_num <- function(df) {
  # create field for start of the week - date of monday of the start of the week
  # convert to week number in the term
  df %>%
    mutate(date = as.Date(time)) %>%
    mutate(
      week = floor(
        difftime(
          time1 = date,
          time2 = term_start,
          units = "weeks"))
    ) %>%
    select(-date)
}

# PREPARING DATA: time elapsed -------------------------------------------------
create_first_last_obs <- function(df) {
  # creating field for first and last obs
  df %>%
    mutate(first_obs = lag(uid) != uid) %>%
    mutate(last_obs = lead(uid) != uid) %>%
    replace_na(list(first_obs = TRUE, last_obs = TRUE))
}

create_time_elapsed <- function(df) {
  # creating time elapsed field (seconds):
  # time_elapsed is calculated as: time - previous time
  df %>%
    arrange(uid, time) %>%  # sort by uid then time
    mutate(time_elapsed = difftime(time, lag(time), units = "mins")) %>%
    replace_na(list(time_elapsed = time_elapsed_impute)) %>%
    # cleanup
    # the lagged time value for the first entry for each uid will lead to
    # non-sensical values. assume the time elapsed for the first and last entries
    # is 20m
    # replacing first obs/last obs (for each uid), NANs and negative with 1200s
    create_first_last_obs() %>%
    replace_na(list(time_elapsed = time_elapsed_impute)) %>%
    # there are non-sensically high values for time elapsed, finding something
    # reasonable to cap it at. Have gone with 30m which is ~98th percentile
    mutate(time_elapsed =
             case_when(
               first_obs                       ~ time_elapsed_impute,
               time_elapsed < 0                ~ time_elapsed_impute,
               time_elapsed > time_elapsed_cap ~ time_elapsed_cap,
               TRUE                            ~ time_elapsed)) %>%
    select(-first_obs, -last_obs)
}

gps <-
  input_data %$%
  gps %>%
  create_week_num() %>%
  create_time_elapsed()

wifi_location <-
  input_data %$%
  wifi_location %>%
  create_week_num() %>%
  create_time_elapsed()


# PREPARING DATA: distance moved -----------------------------------------------
gps <-
  gps %>%
  create_first_last_obs() %>%
  mutate(dist_moved_long = longitude - lag(longitude)) %>%
  mutate(dist_moved_lat = latitude - lag(latitude)) %>%
  mutate(dist_moved = sqrt(dist_moved_long ^ 2 + dist_moved_lat ^ 2)) %>%
  # if it's the first obs, zero it out as the previous records is from another
  # uid
  mutate(dist_moved = dist_moved * 0 ^ first_obs) %>%
  replace_na(list(dist_moved = 0)) %>%
  select(-one_of(c("first_obs", "last_obs", "dist_moved_lat",
                   "dist_moved_long"))) %>%
  # capping dist_moved at the value of the 98th percentile to remove
  # non-sensical values
  mutate(dist_moved = pmin(dist_moved, distance_moved_cap))

# FEATURES: speed -----------------------------------------------------------
# average speed in each week
speed_mean <-
  gps %>%
  group_by(uid, week) %>%
  summarise(speed_mean = mean(speed, na.rm = TRUE))

# max speed in each week
speed_max <-
  gps %>%
  group_by(uid, week) %>%
  summarise(speed_max = max(speed))

# standard deviation of speed in each week
speed_sd <-
  gps %>%
  group_by(uid, week) %>%
  summarise(speed_sd = sd(speed, na.rm = TRUE))

# combine features together
features_wifi_gps <- tibble() %>%
  spread_join(speed_mean, "speed_mean",  TRUE) %>%
  spread_join(speed_max, "speed_max") %>%
  spread_join(speed_sd, "speed_sd")


# FEATURES: travelstate --------------------------------------------------------
# total time spent stationary every week
travelstate_time_stationary <-
  gps %>%
  filter(travelstate == "stationary") %>%
  group_by(uid, week) %>%
  summarise(travelstate_time_stationary = sum(time_elapsed, na.rm = TRUE))

# total time spent moving every week
travelstate_time_moving <-
  gps %>%
  filter(travelstate == "moving") %>%
  group_by(uid, week) %>%
  summarise(travelstate_time_moving = sum(time_elapsed, na.rm = TRUE))

features_wifi_gps <-
  features_wifi_gps %>%
  spread_join(travelstate_time_stationary, "travelstate_time_stationary") %>%
  spread_join(travelstate_time_moving, "travelstate_time_moving")

# FEATURES: indoor/outdoor -----------------------------------------------------
# the paper uses gps to proxy time spent outdoors
outdoor_time <-
  gps %>%
  group_by(uid, week) %>%
  filter(provider == "gps") %>%
  summarise(outdoor_time = sum(time_elapsed, na.rm = TRUE))

# time spent indoors using provider == "gps" as a proxy
indoor_time <-
  gps %>%
  group_by(uid, week) %>%
  filter(provider %in% c("network", "fused")) %>%
  summarise(indoor_time = sum(time_elapsed, na.rm = TRUE))

# distance moved outdoors using provider == "gps" as a proxy
outdoors_dist <-
  gps %>%
  group_by(uid, week) %>%
  filter(provider == "gps") %>%
  summarise(outdoors_dist = sum(dist_moved, na.rm = TRUE))

# distance moved indoors using provider == "gps" as a proxy
indoor_dist <-
  gps %>%
  group_by(uid, week) %>%
  filter(provider %in% c("network", "fused")) %>%
  summarise(indoor_dist = sum(dist_moved, na.rm = TRUE))

features_wifi_gps <-
  features_wifi_gps %>%
  spread_join(outdoor_time, "outdoor_time") %>%
  spread_join(indoor_time, "indoor_time") %>%
  spread_join(indoor_dist, "indoor_dist") %>%
  spread_join(outdoors_dist, "outdoors_dist")

# FEATURES : altitude ----------------------------------------------------------
# mean altitude in each week
altitude_mean <-
  gps %>%
  group_by(uid, week) %>%
  summarise(altitude_mean = mean(altitude, na.rm = TRUE))

# standard deviation of altitude in each week
altitude_sd <-
  gps %>%
  group_by(uid, week) %>%
  summarise(altitude_sd = sd(altitude, na.rm = TRUE))

# max of altitude in each week
altitude_max <-
  gps %>%
  group_by(uid, week) %>%
  summarise(altitude_max = max(altitude, na.rm = TRUE))

# min of altitude in each week
altitude_min <-
  gps %>%
  group_by(uid, week) %>%
  summarise(altitude_min = min(altitude, na.rm = TRUE))

features_wifi_gps <-
  features_wifi_gps %>%
  spread_join(altitude_mean, "altitude_mean") %>%
  spread_join(altitude_sd,   "altitude_sd") %>%
  spread_join(altitude_max,  "altitude_max") %>%
  spread_join(altitude_min,  "altitude_min")

# FEATURES : locations ---------------------------------------------------------
# number of different locations visited each week
location_count <-
  wifi_location %>%
  group_by(uid, week, location) %>%
  tally() %>%
  group_by(uid, week) %>%
  tally(name = "location_count")

features_wifi_gps <- spread_join(features_wifi_gps, location_count, "location_count")

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

common_locations_mapping = c("in[north-main]" = "location_1",
                             "in[sudikoff]"   = "location_2",
                             "in[mclaughlin]" = "location_3",
                             "in[north-park]" = "location_4",
                             "in[massrow]"    = "location_5")

location_time <-
  wifi_location %>%
  filter(location %in% names(common_locations_mapping)) %>%
  mutate(location = common_locations_mapping[location]) %>%
  group_by(uid, week, location) %>%
  summarise(location_time = sum(time_elapsed, na.rm = TRUE)) %>%
  ungroup() %>%
  spread(location, location_time, fill = 0) %>%
  rename_at(vars(starts_with("location_")), ~ stri_c(., "_time")) %>%
  mutate_if(is.difftime, as.numeric)


for (i in seq_len(length(common_locations_mapping))) {
  # dplyr magic required to work with lazy evaluation
  location_time_col <- stri_c("location_", i, "_time")
  location_time_col <- enquo(location_time_col)
  features_wifi_gps <-
    features_wifi_gps %>%
    spread_join(
      location_time %>%
        select(uid, week, !!location_time_col),
      stri_c("location_", i, "_time"))
}

# FEATURES : bearing -----------------------------------------------------------
# only valid if network is GPS
# group into N E S W

bearing_time <-
  gps %>%
  filter(provider == "gps") %>%
  mutate(bearing = round(bearing / 90) %% 4) %>%
  mutate(bearing =
           case_when(
             bearing == 0 ~ "north",
             bearing == 1 ~ "east",
             bearing == 2 ~ "south",
             bearing == 3 ~ "west",
             TRUE         ~ "error"
           )
  ) %>%
  group_by(uid, week, bearing) %>%
  summarise(bearing_time = sum(time_elapsed, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate_if(is.difftime, as.numeric)

for (direction in c("north", "east", "south", "west")) {
  bearing_time_col <- stri_c("bearing_", direction, "_time")
  bearing_time_col <- enquo(bearing_time_col)
  features_wifi_gps <- spread_join(features_wifi_gps,
                                   bearing_time %>%
                                     filter(bearing == direction) %>%
                                     select(-bearing) %>%
                                     rename(!!bearing_time_col := bearing_time),
                                   stri_c("bearing_", direction, "_time"))
}

# FEATURES : export ------------------------------------------------------------
write_csv(features_wifi_gps, file.path(feature_folder, feature_name))
