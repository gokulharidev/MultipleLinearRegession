import numpy as np

class LinearRegression:
    def __init__(self, delta, alpha=0.1, iterations=1000, lr=0.01):
        self.alpha = alpha
        self.epoch = iterations
        self.w = None
        self.b = 0
        self.lr = lr
        self.delta = delta
    
    def huber_loss(self, error):
        return np.where(np.abs(error) <= self.delta, error, np.sign(error) * self.delta)
    
    def gradient_descent(self, x, y):
        y_pred = np.dot(x, self.w) + self.b
        error = y - y_pred
        grad = self.huber_loss(error)
        
        dw = (1 / x.shape[0]) * np.dot(x.T, grad)
        db = (1 / x.shape[0]) * np.sum(grad)
        
        self.w -= self.alpha * dw * self.lr
        self.b -= self.alpha * db * self.lr
        
    def fit(self, x, y):
        self.w = np.zeros(x.shape[1])
        for _ in range(self.epoch):
            self.gradient_descent(x, y)
    
    def predict(self, x):
        return np.dot(x, self.w) + self.b
    
    def r_square(self, y, y_pred):
        ss_res = np.sum((y - y_pred) ** 2)  # Fixed squaring issue
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def tune_delta(self, x, y):
        deltas = [0.1, 0.5, 1, 2, 5, 10]
        best_score = -np.inf
        best_delta = None

        for delta in deltas:
            self.delta = delta
            self.fit(x, y)
            y_pred = self.predict(x)
            score = self.r_square(y, y_pred)

            if score > best_score:
                best_score = score
                best_delta = delta
            
            print(f"Delta: {delta}, R^2: {score}")

        return best_delta
