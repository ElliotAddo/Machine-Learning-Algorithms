library(readxl)
library(ggplot2)
library(gridExtra)


# Plot 1
plot1 <- ggplot(ACCESS, aes(x = 1:nrow(ACCESS), group = 1)) +
  geom_line(aes(y = `Actual`, color = "Actual"), size = 1) +
  geom_line(aes(y = `Linear Regression`, color = "Linear Regression"), size = 1) +
  labs(title = "Actual vs Linear Regression",x = "Days", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Linear Regression" = "green")) +
  theme_minimal() + scale_x_continuous(breaks = seq(1, nrow(ACCESS), by = 100),labels = seq(1, nrow(ACCESS), by = 100))


# Plot2 
plot2 <- ggplot(ACCESS, aes(x = 1:nrow(ACCESS), group = 1)) +
  geom_line(aes(y = `Actual`, color = "Actual"), size = 1) +
  geom_line(aes(y = `Neural Network`, color = "Neural Network"), size = 1) +
  labs(title = "Actual vs Neural Network", x = "Days",y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Neural Network" = "blue")) +
  theme_minimal() + scale_x_continuous(breaks = seq(1, nrow(ACCESS), by = 100),labels = seq(1, nrow(ACCESS), by = 100))

# Plot 3
plot3 <- ggplot(ACCESS, aes(x = 1:nrow(ACCESS), group = 1)) +
  geom_line(aes(y = `Actual`, color = "Actual"), size = 1) +
  geom_line(aes(y = `Random Forest`, color = "Random Forest"), size = 1) +
  labs(title = "Actual vs Random Forest",x = "Days", y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Random Forest" = "purple")) + theme_minimal() +
  scale_x_continuous(breaks = seq(1, nrow(ACCESS), by = 100),
                     labels = seq(1, nrow(ACCESS), by = 100))


# Plot 4
plot4 <- ggplot(ACCESS, aes(x = 1:nrow(ACCESS), group = 1)) +
  geom_line(aes(y = `Actual`, color = "Actual"), size = 1) +
  geom_line(aes(y = `kNN`, color = "kNN"), size = 1) +
  labs(title = "Actual vs kNN", x = "Days",y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "kNN" = "yellow")) +
  theme_minimal() + scale_x_continuous(breaks = seq(1, nrow(ACCESS), by = 100),
                     labels = seq(1, nrow(ACCESS), by = 100))

# Plot 5
plot5 <- ggplot(ACCESS, aes(x = 1:nrow(ACCESS), group = 1)) +
  geom_line(aes(y = `Actual`, color = "Actual"), size = 1) +
  geom_line(aes(y = `Tree`, color = "Tree"), size = 1) +
  labs(title = "Actual vs Tree", x = "Days",y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "Tree" = "orange")) +
  theme_minimal() + scale_x_continuous(breaks = seq(1, nrow(ACCESS), by = 100),
                     labels = seq(1, nrow(ACCESS), by = 100))



#plot6

plot6 <- ggplot(ACCESS, aes(x = 1:nrow(ACCESS), group = 1)) +
  geom_line(aes(y = `Actual`, color = "Actual"), size = 1) +
  geom_line(aes(y = `SVM`, color = "SVM"), size = 1) +
  labs(title = "Actual vs SVM",x = "Days",y = "Values") +
  scale_color_manual(values = c("Actual" = "red", "SVM" = "brown")) +
  theme_minimal() + scale_x_continuous(breaks = seq(1, nrow(ACCESS), by = 100),
                     labels = seq(1, nrow(ACCESS), by = 100))


# Combine plots
grid.arrange(plot1, plot2, plot3, plot4, plot5,plot6, ncol = 2)

