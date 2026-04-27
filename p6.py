import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.2, 100)
tau = 0.5

def lwr(x_q, X, y, tau):
    weights = np.exp(-(X - x_q)**2 / (2 * tau**2))
    A = np.column_stack([weights, weights * X])
    theta = np.linalg.lstsq(A, weights * y, rcond=None)[0]
    return [1, x_q] @ theta

y_pred = [lwr(x, X, y, tau) for x in X]

plt.scatter(X, y, s=10)
plt.plot(X, y_pred, 'r', label=f'tau={tau}')
plt.legend()	
plt.show()