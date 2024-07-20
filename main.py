import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
def gradient_descent(x1: list[float], 
                     x2: list[float], 
                     x3: list[float], 
                     x4: list[float], 
                     y: list[int]):
    
    w1 = w2 = w3 = w4 = b = 0
    iterations = 10000
    n = len(x1)
    learning_rate = 0.0001
    costs = []
    for i in range(iterations):
        y_predicted = w1*x1 + w2*x2 + w3*x3 + w4*x4 + b
        cost =  (1/n) * sum([val**2 for val in (y-y_predicted)])
        costs.append(cost)
        grad_1 = -(2/n)*sum(x1*(y-y_predicted))
        grad_2 = -(2/n)*sum(x2*(y-y_predicted))
        grad_3 = -(2/n)*sum(x3*(y-y_predicted))
        grad_4 = -(2/n)*sum(x4*(y-y_predicted))
        grad_5 = -(2/n)*sum(y-y_predicted)

        w1 = w1 - learning_rate * grad_1
        w2 = w2 - learning_rate * grad_2
        w3 = w3 - learning_rate * grad_3
        w4 = w4 - learning_rate * grad_4
        b = b - learning_rate * grad_5

        y_predicted = [math.ceil(x) for x in y_predicted]
        print(cost)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(iterations), costs, 'b-')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function during Gradient Descent')
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, c='blue', label='Actual')
    ax.set_xlabel('SepalLengthCm')
    ax.set_ylabel('SepalWidthCm')
    ax.set_zlabel('Target Value')

    x1_range = np.linspace(min(x1), max(x1), 10)
    x2_range = np.linspace(min(x2), max(x2), 10)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Y_pred = w1*X1 + w2*X2 + w3*2 + w4*2 + b  # Assuming fixed x3 and x4 values

    ax.plot_surface(X1, X2, Y_pred, color='red', alpha=0.5)

    plt.title('3D Plot of Actual vs Predicted Values')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("Iris.csv")

    x1 = df["SepalLengthCm"]
    x2 = df["SepalWidthCm"]
    x3 = df["PetalLengthCm"]
    x4 = df["PetalWidthCm"]

    dct = {"Iris-setosa": 0, "Iris-versicolor": 2, "Iris-virginica": 4}
    y = df["Species"].apply(lambda x: dct[x])

    gradient_descent(x1, x2, x3, x4, y)