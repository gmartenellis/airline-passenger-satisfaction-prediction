# Airline Passenger Satisfaction - Naive Bayes (Geraldine Marten-Ellis)
# DS520 Data Mining Team Project - 89.11% Accuracy
# Datasets: Download from Kaggle 
# train.csv: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

# Load required packages
library(caret)
library(tidyverse)
library(naivebayes)

# Load datasets (update paths to your files)
df_train <- read.csv("train.csv", stringsAsFactors = FALSE)
df_test <- read.csv("test.csv", stringsAsFactors = FALSE)

# Data preprocessing - My contribution
df_train <- df_train %>%
  select(-X, -id, -Departure.Delay.in.Minutes) %>%
  mutate(
    satisfaction = as.factor(satisfaction),
    Gender = as.factor(Gender),
    Customer.Type = as.factor(Customer.Type),
    Type.of.Travel = as.factor(Type.of.Travel),
    Class = as.factor(Class)
  )

df_test <- df_test %>%
  select(-X, -id, -Departure.Delay.in.Minutes) %>%
  mutate(
    satisfaction = as.factor(satisfaction),
    Gender = as.factor(Gender),
    Customer.Type = as.factor(Customer.Type),
    Type.of.Travel = as.factor(Type.of.Travel),
    Class = as.factor(Class)
  )

# Check data
summary(df_train)
table(df_train$satisfaction)

# Train Naive Bayes with 10-fold CV - My core contribution
set.seed(520)
nb_model <- train(
  satisfaction ~ ., 
  data = df_train,
  method = "naive_bayes",
  trControl = trainControl(method = "cv", number = 10)
)

# Predict & evaluate
predictions <- predict(nb_model, newdata = df_test)
confusion <- confusionMatrix(predictions, df_test$satisfaction)
print(confusion)

# Results: 89.11% accuracy
print(nb_model$bestTune)
