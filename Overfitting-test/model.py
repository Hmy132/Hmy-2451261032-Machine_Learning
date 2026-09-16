from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def create_model(degree):
    return make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )