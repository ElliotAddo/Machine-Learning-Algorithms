library(readr)
library(caret) #for predictive models and contains ggplot2 for visualization
library(e1071)# for SVM
library(tidyverse)#contains lubridate, used when dealing with dates
library(rpart) #for decision tree
library(randomForest)  # for random forest


# Loading the data set
MTN <- read_csv("C:/Users/Elliot/OneDrive/Desktop/Predicted/MTN.csv")

# Scaling the data to a [0,1] range
scaler <- preProcess(MTN[, c("Open", "High", "Low", "Close")], method = "range")
scaled_data <- predict(scaler, MTN[, c("Open", "High", "Low", "Close")])

# Splitting the data into independent and dependent variables
x <- as.matrix(scaled_data[, c("Open", "High", "Low")])
y <- as.vector(scaled_data$Close)

# Splitting the data into training(80%) and testing sets(20%)
set.seed(42)
trainIndex <- createDataPartition(y, p = .8,list = FALSE, times = 1)

X_train <- x[trainIndex, ]
X_train

X_test <- x[-trainIndex, ]
X_test

y_train <- y[trainIndex]
y_train

y_test <- y[-trainIndex]
y_test

# Training the models
# KNN
kNN_model <- train(X_train, y_train, method = "knn", tuneLength = 5)

# SVM
svm_model <- svm(X_train, y_train, kernel = "radial")

# Decision Tree
dt_model <- rpart(y_train ~ ., data = as.data.frame(X_train), method = "anova")

# Random Forest
rf_model <- randomForest(X_train, y_train)


# Making predictions
# KNN
kNN_pred <- predict(kNN_model, X_test)
kNN_pred

# SVM
svm_pred <- predict(svm_model, X_test)
svm_pred

# Decision Tree
dt_pred <- predict(dt_model, as.data.frame(X_test))
dt_pred

# Random Forest
rf_pred <- predict(rf_model, X_test)
rf_pred

# Inverse transform of the predictions to get actual values
inverse_transform <- function(scaled_data, pred) { temp <- as.data.frame(cbind(X_test, pred))
  colnames(temp) <- colnames(scaled_data)
  temp <- predict(scaler, temp, inverse = TRUE)
  return(temp$Close)
}

kNN_pred_actual <- inverse_transform(scaled_data, kNN_pred)
svm_pred_actual <- inverse_transform(scaled_data, svm_pred)
dt_pred_actual <- inverse_transform(scaled_data, dt_pred)
rf_pred_actual <- inverse_transform(scaled_data, rf_pred)
y_test_actual <- inverse_transform(scaled_data, y_test)


#PLOTTING THE VARIOUS MODELS
plot_data <- data.frame(Day = 1:length(y_test_actual),Actual = y_test_actual, KNN = kNN_pred_actual,SVM = svm_pred_actual,DT = dt_pred_actual,RF = rf_pred_actual)

plot_data_long <- pivot_longer(plot_data, cols = c(Actual, KNN, SVM, DT, RF), 
                               names_to = "Model", values_to = "Close_Price")

ggplot(plot_data_long, aes(x = Day, y = Close_Price, color = Model)) +
  geom_line() +
  labs(title = "Actual vs Predicted Close Price", x = "Day", y = "Close Price") +
  theme_minimal() +
  scale_color_manual(values = c("black", "orange", "blue", "green", "purple"))


# Calculating RMSE for each model
kNN_rmse <- RMSE(kNN_pred_actual, y_test_actual)
kNN_rmse

svm_rmse <- RMSE(svm_pred_actual, y_test_actual)
svm_rmse

dt_rmse <- RMSE(dt_pred_actual, y_test_actual)
dt_rmse

rf_rmse <- RMSE(rf_pred_actual, y_test_actual)
rf_rmse



