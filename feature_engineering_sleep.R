library(sqldf)
# the irony of losing sleep to estimate how much sleep other uni students
# got should not be lost on anyone

# authors of study used a statistical model to identify sleep but don't
# push the coefficients or thresholds

# we will take a best guess

# make life easy by copying relevant tibbles

# sleep duration

sleep_coefs <-
  c("dark" = 0.0415,
    "lock" = 0.0512,
    "charge" = 0.0469,
    "stationary" = 0.5445,
    "silence" = 0.3484)


dark <- input_data$dark
phonelock <- input_data$phonelock
phonecharge <- input_data$phonecharge
activity <- input_data$activity
audio <- input_data$audio

activity <-
  activity %>%
  mutate(timestamp = round_date(timestamp, "30 minutes")) %>%
  group_by(uid, timestamp) %>%
  summarise(stationary = sum(activity_inference == "stationary")/ n())

audio <-
  audio %>%
  mutate(timestamp = round_date(timestamp, "30 minutes")) %>%
  group_by(uid, timestamp) %>%
  summarise(silence = sum(audio_inference == "silence")/ n())

start_min <-
  min(dark$start, phonelock$start, phonecharge$start) %>%
  floor_date("30 minutes")
end_max <-
  max(dark$end, phonelock$end, phonecharge$end) %>%
  ceiling_date("30 minutes")

uids <- dark %>% pull(uid) %>% unique()
# 60 seconds * 30 minutes
timestamps <- seq(start_min, end_max, by = 60 * 30)

base_table <- crossing(uid = uids, timestamp = timestamps)
# this is very memory intensive (takes about 16GB of ram)
dark <-
  sqldf(
    "select
      base_table.*
      ,start
      ,end
    from
      base_table
    left join
      dark x
    on
      x.uid = base_table.uid
      and x.start <= base_table.timestamp
      and x.end   >= base_table.timestamp"
    ) %>%
  as_tibble() %>%
  mutate(dark = 1*!is.na(start)) %>%
  select(-start, -end) %>%
  group_by(uid, timestamp) %>%
  summarise(dark = max(dark)) %>%
  ungroup()

phonelock <-
  sqldf(
    "select
      base_table.*
      ,start
      ,end
    from
      base_table
    left join
      phonelock x
    on
      x.uid = base_table.uid
      and x.start <= base_table.timestamp
      and x.end   >= base_table.timestamp"
  ) %>%
  as_tibble() %>%
  mutate(phonelock = 1*!is.na(start)) %>%
  select(-start, -end) %>%
  group_by(uid, timestamp) %>%
  summarise(phonelock = max(phonelock)) %>%
  ungroup()

phonecharge <-
  sqldf(
    "select
      base_table.*
      ,start
      ,end
    from
      base_table
    left join
      phonecharge x
    on
      x.uid = base_table.uid
      and x.start <= base_table.timestamp
      and x.end   >= base_table.timestamp"
  ) %>%
  as_tibble() %>%
  mutate(phonecharge = 1*!is.na(start)) %>%
  select(-start, -end) %>%
  group_by(uid, timestamp) %>%
  summarise(phonecharge = max(phonecharge)) %>%
  ungroup()

sleep <-
  dark %>%
  left_join(phonelock, by = c("uid", "timestamp")) %>%
  left_join(phonecharge, by = c("uid", "timestamp")) %>%
  left_join(activity, by = c("uid", "timestamp")) %>%
  left_join(audio, by = c("uid", "timestamp"))

date_term_start <- ymd("20130318", tz = "US/Eastern")

sleep <-
  sleep %>%
  mutate(week_day = wday(timestamp, label = TRUE, abbr = FALSE)) %>%
  mutate(sleep_hours =
           # 0.5 is used since we have a half hour block
           0.5 * (
             sleep_coefs["dark"] * dark
           + sleep_coefs["lock"] * phonelock
           + sleep_coefs["charge"] * phonecharge
           + sleep_coefs["stationary"] * stationary
           + sleep_coefs["silence"] * silence)) %>%
  # some basic cleaning to account for cases where we over-estiamte sleep
  mutate(sleep_hours = if_else((dark + phonelock + phonecharge + stationary + silence) < 3, 0, sleep_hours )) %>%
  mutate(sleep_hours = if_else(sleep_hours < 0.25, 0, sleep_hours)) %>%
  # -1 is used so we start at 1 not 0
  mutate(week_num = as.integer(ceiling(difftime(as.Date(timestamp), as.Date(start_min) - 1, units = "weeks"))))

# summarise sleep information and create weekly features
sleep <-
  sleep %>%
  group_by(uid, week_num, week_day) %>%
  summarise(sleep_hours = sum(sleep_hours, na.rm = TRUE)) %>%
  group_by(uid, week_num) %>%
  summarise(sleep_mean = sum(sleep_hours, na.rm = TRUE),
            sleep_max   = max(sleep_hours, na.rm = TRUE),
            sleep_min   = min(sleep_hours, na.rm = TRUE),
            sleep_med   = median(sleep_hours, na.rm = TRUE)) %>%
  ungroup() %>%
  gather(variable, value, -(uid:week_num)) %>%
  mutate(variable = stri_c(variable, "_wk")) %>%
  unite(temp, variable, week_num) %>%
  spread(temp, -uid)

# write to csv
write_csv(sleep, "preprocessed_data/features_sleep.csv")
