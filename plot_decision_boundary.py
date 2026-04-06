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


def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(X, y, model)