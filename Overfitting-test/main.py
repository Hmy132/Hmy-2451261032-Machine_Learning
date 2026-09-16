from data import create_dataset
from evaluation import evaluate_model
from model import create_model

df = create_dataset()

X = df[["area", "rooms", "distance"]]
y = df["price"]

for degree in [1, 2, 3, 5, 10, 15]:

    model = create_model(degree)

    mse = evaluate_model(model, X, y)

    print(f"Degree {degree}: CV MSE = {mse:.4f}")