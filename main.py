import pandas as pd
import math

def gradient_descent(x1: list[float], 
                     x2: list[float], 
                     x3: list[float], 
                     x4: list[float], 
                     y: list[int]):
    
    w1 = w2 = w3 = w4 = b = 0
    iterations = 10000
    n = len(x1)
    learning_rate = 0.0001

    for i in range(iterations):
        y_predicted = w1*x1 + w2*x2 + w3*x3 + w4*x4 + b
        cost =  (1/n) * sum([val**2 for val in (y-y_predicted)])

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

if __name__ == "__main__":
    df = pd.read_csv("Iris.csv")

    x1 = df["SepalLengthCm"]
    x2 = df["SepalWidthCm"]
    x3 = df["PetalLengthCm"]
    x4 = df["PetalWidthCm"]

    dct = {"Iris-setosa": 0, "Iris-versicolor": 2, "Iris-virginica": 4}
    y = df["Species"].apply(lambda x: dct[x])

    gradient_descent(x1, x2, x3, x4, y)