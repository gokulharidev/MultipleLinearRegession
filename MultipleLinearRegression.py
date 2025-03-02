import numpy as np

class LinearRegression:
    def __init__(self, delta=1.0, alpha=0.1, iterations=1000, lr=0.01, tolerance=1e-6):
        self.alpha = alpha
        self.epoch = iterations
        self.lr = lr
        self.delta = delta
        self.tolerance = tolerance
        self.w = None
        self.b = 0
    
    def huber_loss(self, error):
        return np.where(np.abs(error) <= self.delta, error, self.delta * np.sign(error))
    
    def gradient_descent(self, x, y):
        
        y_pred = np.dot(x, self.w) + self.b
        error = y - y_pred
        grad = self.huber_loss(error)

        dw = (1 / x.shape[0]) * np.dot(x.T, grad) + self.alpha * self.w 
        db = (1 / x.shape[0]) * np.sum(grad)

        dw += self.alpha * np.sign(self.w)

        self.w -= self.lr * dw
        self.b -= self.lr * db  

        return np.mean(error ** 2),y_pred
    
    def fit(self, x, y):
    
        np.random.seed(42)
        self.w = np.random.randn(x.shape[1]) * 0.01 

        prev_loss = float('inf')
        for i in range(self.epoch):
            loss,y_pred = self.gradient_descent(x, y)

            if abs(prev_loss - loss) < self.tolerance:
                print(f"Early stopping at epoch {i+1} (Loss: {loss:.6f})")
                break
        print("VIF for multicollinearity:", self.VIF(y,y_pred))
        prev_loss = loss
    
    def predict(self, x):
        return np.dot(x, self.w) + self.b
    
    def r_square(self, y, y_pred):
        ss_res = np.sum((y - y_pred) ** 2)
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
            
            print(f"Delta: {delta}, R^2: {score:.4f}")

        return best_delta

    def VIF(self,y,y_pred):
        r=self.r_square(y, y_pred)
        vif = 1/(1-r)
        return vif