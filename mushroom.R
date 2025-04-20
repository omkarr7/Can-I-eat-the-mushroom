rm(list=ls())

library(stats)
library(dplyr)
library(corrplot)
library(ggplot2)
data <- read.csv('./mushrooms.csv')

sum(is.na(df))
head(df)

df <- data %>%
  mutate(
    Edible = as.numeric(factor(Edible)),
    CapShape = as.numeric(factor(CapShape)),
    CapSurface = as.numeric(factor(CapSurface)),
    CapColor = as.numeric(factor(CapColor)),
    Odor = as.numeric(factor(Odor)),
    Height = as.numeric(factor(Height))
  )

cor_matrix <- cor(df)

# Plot the correlation matrix using corrplot
corrplot(cor_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 45)

heatmap_data <- df %>%
  count(CapShape, CapColor) %>%
  spread(key = CapColor, value = n, fill = 0)

# Convert the data to long format for ggplot
heatmap_data_long <- heatmap_data %>%
  gather(key = "CapColor", value = "Count", -CapShape)

# Create the heatmap
ggplot(heatmap_data_long, aes(x = CapShape, y = CapColor, fill = Count)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  ggtitle("Heatmap of Cap Shape vs. Cap Color") +
  xlab("Cap Shape") +
  ylab("Cap Color") +
  theme_minimal()

# Fit the logistic regression model
model <- glm(Edible ~ CapShape + CapSurface + CapColor + Odor + Height, data = data, family = binomial)

# Define the grid for beta_0 and beta_1
B0 <- seq(-5, 5, length.out = 100)
B1 <- seq(0, 5, length.out = 101)

# Define an empty matrix for log-likelihood values
LL <- matrix(NA, 100, 101)

# Function to calculate the log-likelihood for given beta_0 and beta_1
log_likelihood <- function(beta_0, beta_1, X, Y) {
  linear_pred <- beta_0 + beta_1 * X
  probabilities <- exp(linear_pred) / (1 + exp(linear_pred))
  ll <- sum(Y * log(probabilities) + (1 - Y) * log(1 - probabilities))
  return(ll)
}

# Assuming the first predictor in the model is CapShapeConvex
X <- ifelse(df$CapShape == "Convex", 1, 0)
Y <- ifelse(df$Edible == "Edible", 1, 0)

# Evaluate the log-likelihood for each pair of B0 and B1
for (i in 1:100) {
  for (j in 1:101) {
    LL[i, j] <- log_likelihood(B0[i], B1[j], X, Y)
  }
}

# Show the results as an image
image(B0, B1, LL, xlab = "Beta 0", ylab = "Beta 1", main = "Log-Likelihood for Beta 0 and Beta 1")

# Find the indices of the maximum log-likelihood value
idx <- which.max(LL)
idx <- arrayInd(idx, dim(LL))

# Extract the corresponding B0 and B1 values
beta_0_mle <- B0[idx[1]]
beta_1_mle <- B1[idx[2]]

# Print the MLE estimates for beta_0 and beta_1
print(paste("MLE estimate for beta 0 is", beta_0_mle))
print(paste("MLE estimate for beta 1 is", beta_1_mle))

# ------------ Task 1 --------------------------------


library(randomForest)
library(rpart)
library(rpart.plot)
set.seed(123)

# Split the data into training (70%) and testing (30%)
train_indices <- sample(1:nrow(df), size = 0.7 * nrow(df))
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]

# Fit the decision tree model
tree_model <- rpart(Edible ~ ., data = train_data, method = "class")

# Plot the decision tree
rpart.plot(tree_model, type = 3, extra = 101)
# Predict on the test set
tree_predictions <- predict(tree_model, test_data, type = "class")

# Calculate the accuracy
tree_accuracy <- sum(tree_predictions == test_data$Edible) / nrow(test_data)
print(paste("Decision Tree Accuracy:", round(tree_accuracy * 100, 2), "%"))

# Fit the random forest model
rf_model <- randomForest(Edible ~ ., data = train_data, ntree = 500, mtry = 2)

# Tune the random forest model using cross-validation (optional)
# Tune the number of trees (ntree) and the number of variables tried at each split (mtry)

# Predict on the test set
rf_predictions <- predict(rf_model, test_data)

# Calculate the accuracy
rf_accuracy <- sum(rf_predictions == test_data$Edible) / nrow(test_data)
print(paste("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%"))

print(paste("Decision Tree Accuracy:", round(tree_accuracy * 100, 2), "%"))
print(paste("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%"))

# Select the best model based on accuracy
if (tree_accuracy > rf_accuracy) {
  best_model <- tree_model
  best_model_name <- "Decision Tree"
} else {
  best_model <- rf_model
  best_model_name <- "Random Forest"
}

print(paste("The best model is:", best_model_name))

if (best_model_name == "Decision Tree") {
  rpart.plot(best_model, type = 3, extra = 101)
  
  # Interpretation
  print("Decision trees are easy to interpret as they provide a clear visual representation of the decision rules.")
  print("Each node in the tree represents a decision based on one of the attributes.")
  print("The paths from the root to the leaves represent different combinations of attribute values leading to the classification.")
} else {
  print("Random forests are generally more accurate but less interpretable compared to decision trees.")
  print("They consist of an ensemble of decision trees, making it hard to visualize the decision-making process.")
}


# ---------- Task 2 -------------------------------

# Load necessary libraries
# install.packages("caret")
# install.packages("e1071")  # Required for some methods in caret
library(caret)

# Define cross-validation settings
control <- trainControl(method = "cv", number = 10, savePredictions = TRUE, classProbs = TRUE)

# Define models to train
models <- list()
models$DecisionTree <- train(Edible ~ ., data = df, method = "rpart", trControl = control)
models$RandomForest <- train(Edible ~ ., data = df, method = "rf", trControl = control)

# Print the results
lapply(models, print)
# Extract accuracy results
results <- resamples(models)

# Summarize the results
summary(results)

# Boxplot to compare the models
bwplot(results, metric = "Accuracy")
# Extract accuracy results
results <- resamples(models)

# Summarize the results
summary(results)

# Boxplot to compare the models
bwplot(results, metric = "Accuracy")
# Perform paired t-test
differences <- diff(results)
t_test_results <- t.test(differences$values[,1], differences$values[,2], paired = TRUE)
print(t_test_results)
# Print the accuracy of both models
cat("Accuracy of Decision Tree:", mean(results$values$DecisionTree, na.rm = TRUE), "\n")
cat("Accuracy of Random Forest:", mean(results$values$RandomForest, na.rm = TRUE), "\n")

# Interpret the t-test results
if (t_test_results$p.value < 0.05) {
  cat("The difference in performance between the Decision Tree and Random Forest models is statistically significant (p-value =", t_test_results$p.value, ")\n")
} else {
  cat("The difference in performance between the Decision Tree and Random Forest models is not statistically significant (p-value =", t_test_results$p.value, ")\n")
}

library(reshape2)
df_melt <- melt(df, id.vars = "Edible")

# Plot density plots using ggplot
ggplot(df_melt, aes(x = value, fill = Edible)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal()

library(ggmosaic)
ggplot(data = df) +
  geom_mosaic(aes(weight = 1, x = product(CapShape), fill = Edible), divider = ddecker()) +
  labs(title = "CapShape by Edible Status", x = "Cap Shape", y = "Proportion")


