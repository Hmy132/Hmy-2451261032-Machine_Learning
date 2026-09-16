import numpy as np
import pandas as pd


def create_dataset(n=50):
    np.random.seed(42)

    area = np.random.randint(40, 150, n)
    rooms = np.random.randint(1, 6, n)
    distance = np.random.uniform(1, 20, n)

    price = (
        0.05 * area
        + 0.5 * rooms
        - 0.03 * distance
        + np.random.normal(0, 2.0, n)
    )

    return pd.DataFrame({
        "area": area,
        "rooms": rooms,
        "distance": distance,
        "price": price
    })