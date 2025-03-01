from MultipleLinearRegression import LinearRegression
import numpy as np

# Generate synthetic dataset with outliers
np.random.seed(42)
X1 = np.random.rand(100, 1)
X2 = 3 * X1 + np.random.normal(0, 0.02, (100, 1))  
X3 = np.random.rand(100, 1)
X = np.hstack((X1, X2, X3))

true_weights = np.array([2, -1, 3])
y = X.dot(true_weights) + np.random.normal(0, 0.1, 100)

# Add some outliers
y[::10] += np.random.normal(10, 5, 10)

# Find the best delta
model = LinearRegression(delta=1)  # Initial model
best_delta = model.tune_delta(X, y)
print("\nBest Delta:", best_delta)

# Train final model with best delta
model = LinearRegression(delta=best_delta)
model.fit(X, y)
y_pred = model.predict(X)

# Print evaluation metrics
print("\nFinal Model Evaluation")
print("R^2:", model.r_square(y, y_pred))
print("Weights:", model.w)
print("Bias:", model.b)