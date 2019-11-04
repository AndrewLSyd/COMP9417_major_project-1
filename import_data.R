library(tidyverse)
library(stringi)
library(data.table)
library(assertthat)
summary.character <- function(object, maxsum, ...) {
  summary(as.factor(object), maxsum, ...)
}

input_sensing_path <- "StudentLife_Dataset/Inputs/sensing"

folders <- list.files(
  path = "StudentLife_Dataset/Inputs/sensing", pattern = "*")

input_files <- list()
input_data <- list()
for (folder in folders) {
  input_files[[folder]] <-list.files(
    path = file.path(input_sensing_path, folder),
    pattern = "*.csv",
    all.files = TRUE, recursive = TRUE,
    include.dirs = TRUE)

  # deal with empty columsn that exist
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

  input_data[[folder]] <-
    input_files[[folder]] %>%
    map_df(~ file.path(input_sensing_path, folder, .x) %>%
             read_csv(col_types = col_types,
                      col_names = col_names,
                      skip = skip) %>%
             mutate(uid = stri_extract_first_regex(.x, "u\\d{2}")) %>%
             # remove empty column we created since it all na
             select(-matches("^empty$")))
}


# challenge: huge amount of variables and very few rows (e.g. )


raw_data <- list()
