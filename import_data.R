# a description of the StudentLife Data set can be found at
# https://studentlife.cs.dartmouth.edu/dataset.html
library(tidyverse)
library(magrittr)
library(lubridate)
library(stringi)
library(assertr)
library(assertthat)
summary.character <- function(object, maxsum, ...) {
  summary(as.factor(object), maxsum, ...)
}

subset_of <- function(...) {
  quietly(one_of)(...)$result
}

clean_column_names <- function(df) {
  old_names <- colnames(df)
  new_names <-
    old_names %>%
    stri_replace_all_fixed(" ", "_") %>%
    stri_replace_all_regex("[^a-zA-Z0-9]", "_") %>%
    stri_replace_all_regex("_+", "_") %>%
    stri_trans_tolower()
  colnames(df) <- new_names
  df
}

input_sensing_path <- "StudentLife_Dataset/Inputs/sensing"

folders <-
  list.files(
    path = input_sensing_path,
    pattern = "*"
  )

input_files <- list()
input_data <- list()
time_cols <- c("time", "start", "end",
               "timestamp", "start_timestamp", "end_timestamp")
for (folder in folders) {
  cat(stri_c("Importing all csv files from ", folder, "\n"))
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
    #clean_column_names removes spaces from column names
    clean_column_names() %>%
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

# import outputs ----------
output_sensing_path <- "StudentLife_Dataset/Outputs"

file_to_name <- function(file_name) {
  file_name %>%
    # camelCase to snake_case
    stri_replace_all_regex("([a-z])([A-Z])", "$1_$2") %>%
    # make everything lower case
    stri_trans_tolower() %>%
    # remove file extension
    stri_replace_all_fixed(".csv", "")
}

output_files <-
  list.files(
    path = output_sensing_path,
    pattern = "*.csv"
  )

# list to store output tibbles
output_data <- list()
for (output_file in output_files) {
  output_data[[file_to_name(output_file)]] <-
    read_csv(file.path(output_sensing_path, output_file),
             col_types = cols()) %>%
    clean_column_names() %>%
    mutate_if(is.double, as.integer)
}

##TODO: agree how to deal with missing values
# main choice:
#  1) remove observations (not recommended we don't have many values)
#  2) mean imputation easy to start off
#  3) row means??
#  4) something smart (e.g. Bayesian PCA as per
#     https://www.omicsonline.org/open-access/a-comparison-of-six-methods-for-missing-data-imputation-2155-6180-1000224.pdf)


output_data[["flourishing_scale"]] <-
  output_data[["flourishing_scale"]] %>%
  rowwise() %>%
  mutate(flourishing_scale =
           i_lead_a_purposeful_and_meaningful_life
         + my_social_relationships_are_supportive_and_rewarding
         + i_am_engaged_and_interested_in_my_daily_activities
         + i_actively_contribute_to_the_happiness_and_well_being_of_others
         + i_am_competent_and_capable_in_the_activities_that_are_important_to_me
         + i_am_a_good_person_and_live_a_good_life
         + i_am_optimistic_about_my_future
         + people_respect_me)

# basic test we have sum all relevant columns
id_cols <- c("uid", "type")
id_cols <- enquo(id_cols)
agg_cols <- c("flourishing_scale")
agg_cols <- enquo(agg_cols)
sum_original <-
  output_data %$%
  flourishing_scale %>%
  select(-(!!id_cols), -(!!agg_cols)) %>%
  sum(na.rm = TRUE)

sum_aggregate <-
  output_data %$%
  flourishing_scale %>%
  select(!!agg_cols) %>%
  sum(na.rm = TRUE)
# this test is expected to fail since we are yet
# to do any imputation and have missing values
assert_that(are_equal(sum_original, sum_aggregate))
rm(id_cols, agg_cols, sum_original, sum_aggregate)

# need to calculate the positive/negative affect score
# excited is missing? according to the PDF there should be a PANAS field called excited?
output_data[["panas"]] <-
  output_data[["panas"]] %>%
  rowwise() %>%
  mutate(panas_pos = sum(interested, strong, enthusiastic, proud, alert,
                         inspired, determined, attentive, active)) %>%
  mutate(panas_neg = sum(distressed, upset, guilty, scared, hostile, irritable,
                         nervous, jittery, afraid)) %>%
  ungroup()

# basic test we have sum all relevant columns
id_cols <- c("uid", "type")
id_cols <- enquo(id_cols)
agg_cols <- c("panas_pos", "panas_neg")
agg_cols <- enquo(agg_cols)
sum_original <-
  output_data %$%
  panas %>%
  select(-(!!id_cols), -(!!agg_cols)) %>%
  sum(na.rm = TRUE)

sum_aggregate <-
  output_data %$%
  panas %>%
  select(!!agg_cols) %>%
  sum(na.rm = TRUE)
# this test is expected to fail since we are yet
# to do any imputation and have missing values
assert_that(are_equal(sum_original, sum_aggregate))
rm(id_cols, agg_cols, sum_original, sum_aggregate)
# keep global environment somewhat clean
rm(col_types, col_names, skip, folder, input_files, time_cols)
