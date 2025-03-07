from enum import Enum

from .. import Triangle


class FitPredict(object):
    
    def __init__(self, model):
        self.model = model

    def fit(self, triangle, **kwargs):
        print(f"Fitting {self.model} to {triangle.name} with config {kwargs}")
        self.fit_triangle = triangle

    def predict(self, triangle = None, **kwargs):
        pred_triangle = triangle or self.fit_triangle 
        print(f"Predicting from {self.model} on {pred_triangle.name} with config {kwargs}")

class DevelopmentModel(Enum):
    ChainLadder = FitPredict("ChainLadder")

class TailModel(Enum):
    GeneralizedBondy = FitPredict("GeneralizedBondy")

class ForecastModel(Enum):
    AR1 = FitPredict("AR1")

