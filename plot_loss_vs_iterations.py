import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

# create synthetic dataset
np.random.seed(42)

X = np.random.randn(200, 2)
y = np.array([1 if x[0] + x[1] > 0 else 0 for x in X])

# train model
model = LogisticRegression(lr=0.1, n_iters=1000)
model.fit(X, y)

# predictions
preds = model.predict(X)

# accuracy
accuracy = np.sum(preds == y) / len(y)
print("Accuracy:", accuracy)


plt.plot(model.losses)
plt.title("Loss vs Iterations")
plt.show()