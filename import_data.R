# a description of the StudentLife Data set can be found at
# https://studentlife.cs.dartmouth.edu/dataset.html
library(tidyverse)
library(lubridate)
library(stringi)
library(assertr)
summary.character <- function(object, maxsum, ...) {
  summary(as.factor(object), maxsum, ...)
}

subset_of <- function(...) {
  quietly(one_of)(...)$result
}

fix_names <- function(df) {
  old_names <- colnames(df)
  new_names <- stri_replace_all_fixed(old_names, " ", "_")
  colnames(df) <- new_names
  df
}

input_sensing_path <- "StudentLife_Dataset/Inputs/sensing"

folders <- list.files(
  path = "StudentLife_Dataset/Inputs/sensing", pattern = "*")

input_files <- list()
input_data <- list()
time_cols <- c("time", "start", "end",
               "timestamp", "start_timestamp", "end_timestamp")
for (folder in folders) {
  input_files[[folder]] <-
    list.files(
      path = file.path(input_sensing_path, folder),
      pattern = "*.csv",
      all.files = TRUE, recursive = TRUE,
      include.dirs = TRUE
    )

  # deal with empty columsn that exist (likely) due to someone not understanding
  # Excel to csv conversio)n
  if (folder == "wifi_location") {
    col_types <-
      cols(
        time = col_double(),
        location = col_character(),
        empty = col_character()
      )
    col_names <- c("time", "location", "empty")
    skip <- 1
  } else if (folder == "gps") {
    col_types <-
      cols(
        time = col_double(),
        provider = col_character(),
        network_type = col_character(),
        accuracy = col_double(),
        latitude = col_double(),
        longitude = col_double(),
        altitude = col_double(),
        bearing = col_double(),
        speed = col_double(),
        travelstate = col_character(),
        empty = col_character()
      )
    col_names <- c("time", "provider", "network_type", "accuracy",
                   "latitude", "longitude", "altitude", "bearing",
                   "speed", "travelstate", "empty")
    skip <- 1
  } else {
    col_types <- cols()
    col_names <- TRUE
    skip <- 0
  }

  #Tthis code is designed for readability over pure speed. Restructing the
  # code to use the data.table package and removing the pipes would likely
  # result in a significant decrease to runtime.
  input_data[[folder]] <-
    input_files[[folder]] %>%
    map_df(~ file.path(input_sensing_path, folder, .x) %>%
             read_csv(col_types = col_types,
                      col_names = col_names,
                      skip = skip) %>%
             mutate(uid = stri_extract_first_regex(.x, "u\\d{2}")) %>%
             # remove empty column we created since it all na
             assert(is.na, matches("^empty$")) %>%
             select(-matches("^empty$"))) %>%
    #fix_names removes spaces from column names
    fix_names() %>%
    # convert linux time stamps into human readable format
    mutate_at(vars(subset_of(time_cols)), as_datetime) %>%
    # convert activity factors into human readable format
    mutate_at(vars(subset_of("activity_inference")),
              ~ .x %>%
                factor(levels = 0:3,
                       labels = c("stationary", "walking", "running", "unknown")) %>%
                as.character()) %>%
    # convert audio factors into human readable format
    mutate_at(vars(subset_of("audio_inference")),
              ~ .x %>%
                factor(levels = 0:3,
                       labels = c("silence", "voice", "noise", "unknown")) %>%
                as.character())

}

# input_data is a list with 10 elements, aeach of with is a data_frame
# corresponding to 1 of the input/sensing subfolders
print(input_data %>% map(summary))

# import outputs
output_sensing_path <- "StudentLife_Dataset/Outputs"

# list to store output tibbles
output_data <- list()

col_types <-
  cols(
    uid = col_character(),
    type = col_character(),
    interested = col_integer(),
    distressed = col_integer(),
    upset = col_integer(),
    strong = col_integer(),
    guilty = col_integer(),
    scared = col_integer(),
    hostile  = col_integer(),
    enthusiastic = col_integer(),
    proud = col_integer(),
    irritable = col_integer(),
    alert = col_integer(),
    inspired = col_integer(),
    nervous = col_integer(),
    determined  = col_integer(),
    attentive = col_integer(),
    jittery = col_integer(),
    active  = col_integer(),
    afraid  = col_integer()
  )

col_names <- c("uid", "type", "interested", "distressed", "upset", "strong", "guilty",
            "scared", "hostile", "enthusiastic", "proud", "irritable", "alert",
            "inspired", "nervous", "determined", "attentive", "jittery", "active",
            "afraid")

output_data[["panas"]] <- read_csv(file.path(output_sensing_path, "panas.csv"),
                                   col_types=col_types, col_names=col_names, skip=1)

# need to calculate the positive/negative affect score
# excited is missing? according to the PDF there should be a PANAS field called excited?
output_data[["panas"]] <- output_data[["panas"]] %>% rowwise() %>%
  mutate(positive_affect= sum(interested, strong , enthusiastic, proud, alert,
                              inspired, determined, attentive, active, na.rm=TRUE))

# afraid is missing? according to the PDF there should be a PANAS field called afraid?
output_data[["panas"]] <- output_data[["panas"]] %>% rowwise() %>%
  mutate(negative_affect = sum(distressed, upset, guilty, scared, hostile, irritable,
                               nervous , jittery, afraid, na.rm=TRUE))

# keep global environment somewhat clean
rm(col_types, col_names, skip, folder, input_files, time_cols)
