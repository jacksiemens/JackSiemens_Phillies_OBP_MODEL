

## XSTATS

library(caret)
library(GeomMLBStadiums)
library(tictoc)
library(baseballr)
library(purrr)
library(lubridate)
library(furrr)
library(foreach)
library(doParallel)
library(pbapply)

# Loading Pitches
seasons <- c(2016, 2017, 2018, 2019, 2020)
schedule_list <- lapply(seasons, function(year) mlb_schedule(season = year, level_ids = "1"))

# Filter each data frame for Regular Season and then extract the date column
schedule_list_filtered <- lapply(schedule_list, function(df) {
  df %>% filter(series_description == "Regular Season") %>% select(date)
})

# Combine all date columns and get the unique dates
all_dates <- do.call(c, lapply(schedule_list_filtered, function(df) df$date))
unique_dates <- unique(all_dates)


# Function to retrieve statcast data for a given date
get_pitches <- function(date) {
  tryCatch({
    statcast_search(start_date = date, end_date = date)
  }, error = function(e) {
    message("Error for date ", date, ": ", e$message)
    return(NULL)  # Return NULL in case of an error
  })
}

# Set up parallel backend with multiple cores
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# Load necessary libraries on the parallel workers
clusterEvalQ(cl, {
  library(baseballr)
  library(tidyverse)
})

# Export the get_pitches function to the parallel workers
clusterExport(cl, "get_pitches")

# Execute the loop in parallel to retrieve data for each date
pitches_list <- foreach(date = unique_dates, .combine = 'c', .packages = "baseballr") %dopar% {
  list(date = date, data = get_pitches(date))
}

# Stop the parallel backend
stopCluster(cl)


# Initialize an empty data frame to store all the successful pitches data
all_pitches <- list()

i <- 2
while (i <= length(pitches_list)) {
  if (is_tibble(pitches_list[i]$data) && nrow(pitches_list[i]$data) > 0) {
    all_pitches[[i / 2]] <- pitches_list[i]$data
  } else {
    warning("The element at index ", i, " is not a tibble.")
  }
  i <- i + 2
}

# Remove NULL elements from the list
all_pitches <- compact(all_pitches)

# Bind all tibbles into one
pitches <- bind_rows(all_pitches)
pitches_16_through_20_ <- pitches


# XSTATS MODELLING
df <- pitches_16_through_20_ %>%
  select(game_year,launch_speed, launch_angle, hc_x, hc_y, events, home_team) %>%
  mlbam_xy_transformation(x = "hc_x", y = "hc_y")%>%
  select(-c(hc_x, hc_y)) %>%
  filter(events != "") %>%
  rename(hc_x = hc_x_, hc_y = hc_y_) %>%
  na.omit()%>%
  mutate(
    spray_angle = atan2(hc_y, hc_x) * (180 / pi)
  ) %>%
  select(-c(hc_y, hc_x))

field_outs <- c("double_play", "field_error", "fielders_choice", "force_out", 
                "grounded_into_double_play", "sac_bunt", "sac_fly", 
                "sac_fly_double_play", "field_out", "fielders_choice_out",
                "triple_play", "sac_bunt_double_play" )

df <- df %>%
  mutate(events = ifelse(events %in% field_outs, "out", events)) %>%
  mutate(events = case_when(
    events == "out" ~ 0, 
    events == "single" ~ 1, 
    events == "double" ~ 2, 
    events == "triple" ~ 3, 
    events == "home_run" ~ 4, 
  ))

df$events <- as.factor(df$events )
df$home_team <- as.factor(df$home_team )


# We shall run this before bed. 
set.seed(123)
training_rows <- sample(nrow(df), 0.3 * nrow(df))
train_data <- df[training_rows, ]
test_data <- df[-training_rows, ]

train_matrix <- model.matrix(events ~ launch_speed + launch_angle + spray_angle + home_team + game_year, train_data)
test_matrix <- model.matrix(events ~ launch_speed + launch_angle + spray_angle + home_team + game_year, test_data)

# Define the control using 5-fold CV
fit_control <- trainControl(method = "cv", number = 5, 
                            summaryFunction = multiClassSummary, 
                            classProbs = TRUE, search = "grid")

# Define the hyperparameter grid to tune
grid <- expand.grid(nrounds = c(100, 150),
                    max_depth = c(3, 5, 7),
                    eta = c(0.01, 0.1, 0.3),
                    gamma = c(0, 0.01),
                    colsample_bytree = c(0.5, 0.7),
                    min_child_weight = c(1, 3, 5),
                    subsample = c(0.5, 0.75, 1.0))

model <- train(train_matrix, train_data$events, method = "xgbTree",
               trControl = fit_control, 
               tuneGrid = grid, 
               metric = "Kappa") 


cv_results_df <- as.data.frame(model$results)
prob_predictions <- predict(model, newdata = test_data, type = "prob")

# Actual class labels
actual_classes <- test_data$events

positive_class_probs <- prob_predictions[, 2]

# Ensure actual_classes is a numeric vector (0 and 1)
actual_classes_numeric <- as.numeric(actual_classes)  # Convert to numeric if not already

# Calculate Brier Score
brier_score <- ModelMetrics::brier(positive_class_probs, actual_classes_numeric)
print(paste("Brier Score:", brier_score))

# Function to plot calibration curve

# Convert the original dataset into a matrix format suitable for xgboost
df_matrix <- model.matrix(~ launch_speed + launch_angle + spray_angle - 1, data = df)

# Make probabilistic predictions on the entire dataset
df_predictions <- predict(model, newdata = df_matrix, type = "prob")

# Convert predictions to a dataframe
df_predictions_df <- as.data.frame(df_predictions)

# Renaming columns to match your requirement
names(df_predictions_df) <- c("pOUT", "p1B", "p2B", "p3B", "pHR")

# Adding these predictions as new columns to the original dataframe
df <- cbind(df, df_predictions_df)


# SUMMARY STATS + EV WORK

xDF <- pitches_16_through_20_ %>%
  select(player_name, batter, game_year,launch_speed, launch_angle, hc_x, hc_y, events, home_team) %>%
  mlbam_xy_transformation(x = "hc_x", y = "hc_y")%>%
  select(-c(hc_x, hc_y)) %>%
  filter(events != "") %>%
  rename(hc_x = hc_x_, hc_y = hc_y_) %>%
  na.omit()%>%
  mutate(
    spray_angle = atan2(hc_y, hc_x) * (180 / pi)
  ) %>%
  select(-c(hc_y, hc_x))

field_outs <- c("double_play", "field_error", "fielders_choice", "force_out", 
                "grounded_into_double_play", "sac_bunt", "sac_fly", 
                "sac_fly_double_play", "field_out", "fielders_choice_out",
                "triple_play", "sac_bunt_double_play" )

xDF <- xDF %>%
  mutate(events = ifelse(events %in% field_outs, "out", events)) %>%
  mutate(events = case_when(
    events == "out" ~ 0, 
    events == "single" ~ 1, 
    events == "double" ~ 2, 
    events == "triple" ~ 3, 
    events == "home_run" ~ 4, 
  ))




df_matrix <- model.matrix(~ launch_speed + launch_angle + spray_angle - 1, data = xDF )

# Make probabilistic predictions on the entire dataset
df_predictions <- predict(model, newdata = df_matrix, type = "prob")

# Convert predictions to a dataframe
df_predictions_df <- as.data.frame(df_predictions)

# Renaming columns to match your requirement
names(df_predictions_df) <- c("pOUT", "p1B", "p2B", "p3B", "pHR")

# Adding these predictions as new columns to the original dataframe
xDF <- cbind(xDF, df_predictions_df)


xDF <- xDF %>%
  mutate(xBACON = (1 - pOUT),
         xSLGCON =  (p1B * 1) + (p2B * 1) + (p3B * 3) + (pHR * 4))%>%
  select(-c(p1B,pOUT,  p2B, p3B )) %>%
  group_by(player_name, batter, game_year) %>%
  summarise(
    BIP = n(),
    xHR = sum(pHR), 
    xBACON = mean(xBACON), 
    xSLGCON = mean(xSLGCON)) %>%
  ungroup() %>%
  arrange(desc(xBACON))
  

EV_metrics <- pitches_16_through_20_ %>%
  filter(!is.na(launch_speed)) %>%
  group_by(player_name, batter, game_year) %>%
  summarize(
    BIP = n(), 
    EV81 = quantile(launch_speed, probs = 0.81),
    EV96 = quantile(launch_speed, probs = 0.96), 
    best_speed = mean(launch_speed[rank(-launch_speed) <= length(launch_speed) / 2]), 
    hard_hit_rate = mean(launch_speed > 95)
  )

player_xStats <- left_join(xDF, EV_metrics, by = c("batter", "game_year")) %>%
  select(-player_name.y, -BIP.y) %>%
  rename(player_name = player_name.x, 
         BIP = BIP.x, 
         Season = game_year, 
         MLBAMID = batter) %>%
  arrange(desc(xHR))

write_csv(player_xStats, "Desktop/R/p_QOC.csv")


# FAILED STUFF MODEL

## Basic Run Value, same as statcast
rv <- pitches_16_through_20_ %>%
  select(pitcher, pitch_type, game_year, delta_run_exp) %>%
  group_by(pitcher, pitch_type, game_year) %>%
  summarize(`RV/100` = round(mean(delta_run_exp) * -100, 2), 
            RV = round(sum(delta_run_exp) * -1, 2), 
            pitches = n())%>%
  ungroup() %>%
  filter(pitches > 100) %>%
  arrange(desc(RV))


# STUFF
# This was a bit of a hail mary, just total black box with very little
# Feature Engineering, I just wanted to see if I could get something decent Quickly
# I did not.

stuff_p <- pitches_16_through_20_ %>%
  select("pitch_type", "release_speed", "release_pos_x" ,"release_pos_z" , "release_pos_y", 
         "p_throws", "stand" ,"pfx_x", "pfx_z",  "plate_x", "plate_z" , 
         "vx0" , "vy0" , "vz0", "ax", "ay","az",  "sz_top" , "sz_bot", "release_spin_rate", 
         "release_extension" ,"spin_axis" , "delta_run_exp")%>%
  na.omit() %>%
  filter(pitch_type %in% c("CH","FF","KC","SI","FC","CU","SV","SL","FS", "SL", "ST" )) %>%
  mutate(stand = ifelse(stand == "R", 0, 1), 
         p_throws = ifelse(p_throws == "R", 0 , 1))

# Label Encoding for 'pitch_type'
unique_pitch_types <- unique(stuff_p$pitch_type)
pitch_type_mapping <- setNames(seq_along(unique_pitch_types), unique_pitch_types)
stuff_p$pitch_type <- as.integer(replace(stuff_p$pitch_type, 
                                         stuff_p$pitch_type %in% names(pitch_type_mapping), 
                                         pitch_type_mapping[stuff_p$pitch_type]))

# Splitting the data into features (X) and target variable (y)
# Set seed for reproducibility
set.seed(123)

# Sample 80% of the data for training
training_rows <- sample(nrow(stuff_p), 0.8 * nrow(stuff_p))
train_data <- stuff_p[training_rows, ]
test_data <- stuff_p[-training_rows, ]

# Define the target variable
target_variable <- "delta_run_exp"

# Create model matrices for training and testing sets
train_matrix <- model.matrix(~. - delta_run_exp, data = train_data)
test_matrix <- model.matrix(~. - delta_run_exp, data = test_data)

# Extract the target variable
train_label <- train_data[[target_variable]]
test_label <- test_data[[target_variable]]

# Convert training and testing data to DMatrix format
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

# XGBoost parameters
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eta = 0.3,
  max_depth = 10,
  subsample = 0.7,
  colsample_bytree = 0.9
)

# Train the model
tic()
stuff_xgb <- xgb.train(params = params, data = dtrain, nrounds = 100)
toc()

# Make predictions on the test set
predictions <- predict(stuff_xgb , dtest)

# Evaluate the model (e.g., using RMSE)
rmse <- sqrt(mean((test_label - predictions)^2))
rmse


calc_r_squared <- function(actual, predicted) {
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r_squared <- 1 - ss_res / ss_tot
  return(r_squared)
}

# Calculate R^2
r_squared <- calc_r_squared(test_label, predictions)
print(paste("R^2:", r_squared))

importance_matrix <- xgb.importance(feature_names = colnames(train_matrix), model = stuff_xgb)
# Plotting Feature Importance
xgb.plot.importance(importance_matrix, top_n = 23, main = "Feature Importance")

  

  





         
         