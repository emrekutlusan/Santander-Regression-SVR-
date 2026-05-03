#install.packages(c("e1071", "caret"))

library(tidyverse)
library(e1071)
library(caret)

# 1. LOAD & PREPARE DATA

df <- read.csv("combined_daily_data.csv") %>%
  filter(borough != "Unknown") %>%
  mutate(date = as.Date(date))


# filter data
df <- df %>%
  filter(!borough %in% c("Wandsworth", "Newham"))


# Select features — total_duration excluded (data leakage)
features <- c(
  "temperature", "humidity", "precipitation",
  "windspeed", "pressure", "cloudcover",
  "Month", "Weekend", "Bank_Holiday",
  "Day_Before_Bank_Holiday", "Day_After_Bank_Holiday",
  "Long_Weekend", "Christmas_Period", "Summer_Period"
)

# Encode categorical: Season, borough, Day_of_Week
df <- df %>%
  mutate(
    Season      = as.integer(factor(Season,
                                    levels = c("Spring","Summer","Autumn","Winter"))),
    borough_enc = as.integer(factor(borough)),
    dow_enc     = as.integer(factor(Day_of_Week,
                                    levels = c("Monday","Tuesday","Wednesday",
                                               "Thursday","Friday","Saturday","Sunday")))
  )

features <- c(features, "Season", "borough_enc", "dow_enc")

cat("Features selected:", length(features), "\n")
cat("  Features:", paste(features, collapse = ", "), "\n\n")

# ============================================================
# 2. TRAIN / TEST SPLIT (2014-2017 train, 2018 test)
#train <- df %>% filter(year(date) < 2018)
#test  <- df %>% filter(year(date) == 2018)
train <- df %>% filter(year(date) < 2018)
test  <- df %>% filter(year(date) == 2018)


cat(" Train size:", nrow(train), "rows\n")
cat("  Test size: ", nrow(test),  "rows\n\n")

X_train <- train[, features]
y_train <- train$trip_count
y_train_log <- log1p(y_train)  # log(1+x), zeros
X_test  <- test[, features]
y_test  <- test$trip_count
y_test_log  <- log1p(y_test)


# 3. SCALE FEATURES (mandatory for SVM)
scaler  <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(scaler, X_train)
X_test_scaled  <- predict(scaler, X_test)

cat(" Features scaled (mean=0, sd=1)\n\n")

# ============================================================
# 4. TRAIN SVM MODEL (RBF kernel)
# ============================================================

cat("Training SVM model...\n")

set.seed(42)
svm_model <- svm(
  x      = X_train_scaled,
  y      = y_train_log,
  type   = "eps-regression",
  kernel = "radial",       # RBF kernel
  cost   = 10,             # regularisation parameter
  gamma  = 0.1,            # RBF bandwidth
  epsilon = 0.1            # epsilon-tube
)

cat(" SVM model trained\n")
cat("  Kernel: Radial (RBF)\n")
cat("  Cost (C):", svm_model$cost, "\n")
cat("  Gamma:", svm_model$gamma, "\n")
cat("  Support vectors:", svm_model$tot.nSV, "\n\n")

# 5. PREDICTIONS & EVALUATION
train_preds <- expm1(predict(svm_model, X_train_scaled))
test_preds  <- expm1(predict(svm_model, X_test_scaled))

# Metrics function
eval_metrics <- function(actual, predicted, label) {
  mae  <- mean(abs(actual - predicted))
  rmse <- sqrt(mean((actual - predicted)^2))
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r2   <- 1 - ss_res / ss_tot
  cat(sprintf("%s — MAE: %.2f | RMSE: %.2f | R²: %.4f\n", label, mae, rmse, r2))
  list(mae = mae, rmse = rmse, r2 = r2)
}

cat("Model Performance:\n")
train_metrics <- eval_metrics(y_train, train_preds, "Train")
test_metrics  <- eval_metrics(y_test,  test_preds,  "Test ")
mape_train <- mean(abs((y_train - train_preds) / y_train)) * 100
mape_test  <- mean(abs((y_test  - test_preds)  / y_test))  * 100
cat(sprintf("Train MAPE: %.2f%%\n", mape_train))
cat(sprintf("Test  MAPE: %.2f%%\n", mape_test))


# 6. RESIDUAL ANALYSIS
residuals_df <- tibble(
  actual    = y_test,
  predicted = test_preds,
  residual  = y_test - test_preds,
  date      = test$date,
  borough   = test$borough
)

# Plot 1: Actual vs Predicted
par(mfrow = c(1, 2))

plot(y_test, test_preds,
     xlab = "Actual Trip Count",
     ylab = "Predicted Trip Count",
     main = "SVM: Actual vs Predicted (Test 2018)",
     pch  = 16, col = rgb(0.2, 0.4, 0.8, 0.4), cex = 0.6)
abline(0, 1, col = "red", lwd = 2, lty = 2)
legend("topleft", legend = c("Predictions", "Perfect fit"),
       col = c(rgb(0.2,0.4,0.8), "red"), pch = c(16, NA),
       lty = c(NA, 2), bty = "n")

# Plot 2: Residuals
plot(test_preds, residuals_df$residual,
     xlab = "Predicted Trip Count",
     ylab = "Residual",
     main = "SVM: Residual Plot",
     pch  = 16, col = rgb(0.8, 0.3, 0.2, 0.4), cex = 0.6)
abline(h = 0, col = "black", lwd = 1.5, lty = 2)

par(mfrow = c(1, 1))


# 8. BOROUGH-LEVEL PERFORMANCE
borough_perf <- residuals_df %>%
  group_by(borough) %>%
  summarise(
    mae  = round(mean(abs(residual)), 2),
    rmse = round(sqrt(mean(residual^2)), 2),
    r2   = round(1 - sum(residual^2) / sum((actual - mean(actual))^2), 4),
    n    = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(r2))

cat("\n Borough-Level Performance (Test 2018):\n")
print(borough_perf)



# 10. SAVE RESULTS
write.csv(residuals_df,  "svm_predictions.csv",    row.names = FALSE)
write.csv(borough_perf,  "svm_borough_perf.csv",   row.names = FALSE)

results_summary <- tibble(
  model  = "SVM (RBF)",
  split  = "Test 2018",
  MAE    = round(test_metrics$mae, 2),
  RMSE   = round(test_metrics$rmse, 2),
  R2     = round(test_metrics$r2, 4)
)
write.csv(results_summary, "svm_results_summary.csv", row.names = FALSE)

cat("\n Done! Files saved:\n")
cat("   svm_predictions.csv\n")
cat("   svm_borough_perf.csv\n")
cat("   svm_results_summary.csv\n")

