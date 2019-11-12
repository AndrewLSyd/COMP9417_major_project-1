# load libraries and define helper functions -----------------------------------
# a description of the StudentLife Data set can be found at
# https://studentlife.cs.dartmouth.edu/dataset.html
library(tidyverse)
library(magrittr)
library(lubridate)
library(stringi)
library(assertr)
library(assertthat)
# the bnstruct package is required but not loaded since it has too many name
# comflict
assert_that("bnstruct" %in% installed.packages())

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

file_to_name <- function(file_name) {
  file_name %>%
    # camelCase to snake_case
    stri_replace_all_regex("([a-z])([A-Z])", "$1_$2") %>%
    # make everything lower case
    stri_trans_tolower() %>%
    # remove file extension
    stri_replace_all_fixed(".csv", "")
}


# import input files -----------------------------------------------------------
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
      all.files = TRUE,
      recursive = TRUE,
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

  # This code is designed for readability over pure speed. Restructing the
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
                       labels = c("stationary", "walking", "running",
                                  "unknown")) %>%
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

# check that results look reasonable
print(input_data %>% map(summary))

# import outputs ---------------------------------------------------------------
output_sensing_path <- "StudentLife_Dataset/Outputs"

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
rm(output_file)

output_data[["flourishing_scale"]] <-
  output_data[["flourishing_scale"]] %>%
  rowwise() %>%
  mutate(flourishing_scale_raw =
           i_lead_a_purposeful_and_meaningful_life
         + my_social_relationships_are_supportive_and_rewarding
         + i_am_engaged_and_interested_in_my_daily_activities
         + i_actively_contribute_to_the_happiness_and_well_being_of_others
         + i_am_competent_and_capable_in_the_activities_that_are_important_to_me
         + i_am_a_good_person_and_live_a_good_life
         + i_am_optimistic_about_my_future
         + people_respect_me) %>%
  ungroup()

# impute missing values
flourishing_scale_matrix <-
  output_data[["flourishing_scale"]] %>%
  select(i_lead_a_purposeful_and_meaningful_life,
         my_social_relationships_are_supportive_and_rewarding,
         i_am_engaged_and_interested_in_my_daily_activities,
         i_actively_contribute_to_the_happiness_and_well_being_of_others,
         i_am_competent_and_capable_in_the_activities_that_are_important_to_me,
         i_am_a_good_person_and_live_a_good_life,
         i_am_optimistic_about_my_future,
         people_respect_me) %>%
  as.matrix()

flourishing_scale_imputated <-
  bnstruct::knn.impute(flourishing_scale_matrix, k = 5, cat.var = numeric(0)) %>%
  rowSums()

output_data[["flourishing_scale"]] <-
  output_data[["flourishing_scale"]] %>%
  mutate(flourishing_scale_imp = flourishing_scale_imputated)

# clean up global environment after imputation
rm(flourishing_scale_matrix, flourishing_scale_imputated)

# basic test we have sum all relevant columns
id_cols <- c("uid", "type")
id_cols <- enquo(id_cols)
agg_raw_cols <- c("flourishing_scale_raw")
agg_raw_cols <- enquo(agg_raw_cols)
agg_imp_cols <- c("flourishing_scale_imp")
agg_imp_cols <- enquo(agg_imp_cols)
sum_original <-
  output_data %$%
  flourishing_scale %>%
  filter(!is.na(flourishing_scale_raw)) %>%
  select(-(!!id_cols), -(!!agg_raw_cols), -(!!agg_imp_cols)) %>%
  sum(na.rm = TRUE)

sum_aggregate_raw <-
  output_data %$%
  flourishing_scale %>%
  # check that all imputed values are not null
  verify(!is.na(flourishing_scale_imp)) %>%
  # check that all imputated values agree to raw values
  verify(is.na(flourishing_scale_raw) | flourishing_scale_imp == flourishing_scale_raw) %>%
  select(!!agg_raw_cols) %>%
  sum(na.rm = TRUE)

# this test is expected to fail since we are yet
# to do any imputation and have missing values
assert_that(are_equal(sum_original, sum_aggregate_raw))
rm(id_cols, sum_original, sum_aggregate_raw)

# need to calculate the positive/negative affect score
# excited and ashamed are missing from the raw data
output_data[["panas"]] <-
  output_data[["panas"]] %>%
  rowwise() %>%
  mutate(panas_pos_raw = sum(interested, strong, enthusiastic, proud, alert,
                             inspired, determined, attentive, active),
         panas_neg_raw = sum(distressed, upset, guilty, scared, hostile,
                             irritable, nervous, jittery, afraid)) %>%
  ungroup()

# impute missing values
panas_scale_matrix <-
  output_data[["panas"]] %>%
  select(interested,
         strong,
         enthusiastic,
         proud,
         alert,
         inspired,
         determined,
         attentive,
         active,
         distressed,
         upset,
         guilty,
         scared,
         hostile,
         irritable,
         nervous,
         jittery,
         afraid) %>%
  mutate_if(is.integer, as.double) %>%
  as.matrix()

panas_scale_imputated <-
  bnstruct::knn.impute(panas_scale_matrix, k = 5, cat.var = numeric(0)) %>%
  as_tibble() %>%
  rowwise() %>%
  mutate(panas_pos_imp = sum(interested, strong, enthusiastic, proud, alert,
                             inspired, determined, attentive, active),
         panas_neg_imp = sum(distressed, upset, guilty, scared, hostile,
                             irritable, nervous, jittery, afraid)) %>%
  ungroup() %>%
  select(panas_pos_imp, panas_neg_imp)

output_data[["panas"]] <-
  output_data[["panas"]] %>%
  mutate(panas_pos_imp = panas_scale_imputated$panas_pos_imp,
         panas_neg_imp = panas_scale_imputated$panas_neg_imp)

rm(panas_scale_matrix, panas_scale_imputated)

# basic test we have sum all relevant columns
id_cols <- c("uid", "type")
id_cols <- enquo(id_cols)
pos_cols <- c("interested", "strong", "enthusiastic", "proud", "alert",
              "inspired", "determined", "attentive", "active")
pos_cols <- enquo(pos_cols)
neg_cols <- c("distressed", "upset", "guilty", "scared", "hostile",
              "irritable", "nervous", "jittery", "afraid")
neg_cols <- enquo(neg_cols)
agg_raw_cols <- c("panas_pos_raw", "panas_neg_raw")
agg_raw_cols <- enquo(agg_raw_cols)
agg_imp_cols <- c("panas_pos_imp", "panas_neg_imp")
agg_imp_cols <- enquo(agg_imp_cols)

sum_original_pos <-
  output_data %$%
  panas %>%
  # check that all imputed values are not null
  verify(!is.na(panas_pos_imp)) %>%
  # check that all imputated values agree to raw values
  verify(is.na(panas_pos_raw) | panas_pos_raw == panas_pos_imp) %>%
  filter(!is.na(panas_pos_raw)) %>%
  select(!!pos_cols) %>%
  sum(na.rm = TRUE)

sum_original_neg <-
  output_data %$%
  panas %>%
  # check that all imputed values are not null
  verify(!is.na(panas_neg_imp)) %>%
  # check that all imputated values agree to raw values
  verify(is.na(panas_neg_raw) | panas_neg_raw == panas_neg_imp) %>%
  filter(!is.na(panas_neg_raw)) %>%
  select(!!neg_cols) %>%
  sum(na.rm = TRUE)


sum_aggregate <-
  output_data %$%
  panas %>%
  select(!!agg_raw_cols) %>%
  sum(na.rm = TRUE)

# this test is expected to fail since we are yet
# to do any imputation and have missing values
assert_that(are_equal(sum_original_pos + sum_original_neg, sum_aggregate))
rm(id_cols, sum_original_pos, sum_original_neg,
   agg_raw_cols, agg_imp_cols, pos_cols, neg_cols, sum_aggregate)
# keep global environment somewhat clean
rm(col_types, col_names, skip, folder, input_files, time_cols)

output_targets <-
  output_data %$%
  left_join(panas, flourishing_scale, by = c("uid", "type")) %>%
  select(uid, type,
         panas_pos_raw, panas_neg_raw, flourishing_scale_raw,
         panas_pos_imp, panas_neg_imp, flourishing_scale_imp)

output_targets <-
  output_targets %>%
  left_join(.,
            mutate_if(., is.numeric, ~ .x >= median(.x, na.rm = TRUE)) %>%
              mutate_if(is.logical, as.integer) %>%
              rename_if(is.integer, list(~stri_c(., "_class"))),
            by = c("uid", "type"))

output_targets %>%
  write_csv("preprocessed_data/target.csv")
