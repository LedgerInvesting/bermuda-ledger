from abc import ABC

import os
from functools import wraps

from .. import Triangle
from .model import DevelopmentModel, TailModel, ForecastModel

Model = DevelopmentModel | TailModel | ForecastModel

class Node(object):

    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.children = []

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.name}"

class TriangleNode(Node):

    def __init__(self, name) -> None:
        super().__init__(name, lambda: None)

class FitNode(Node):

    def __init__(self, name, model, **config) -> None:
        function = model.value
        self.config = config
        super().__init__(name, function)

    def __call__(self, *args, **kwargs):
        self.function.fit(*args, **self.config, **kwargs)
        return self

class PredictNode(Node):

    def __init__(self, name, model, **config) -> None:
        self.config = config
        super().__init__(name, model)

    def __call__(self, *args, **kwargs):
        self.triangle = self.function.predict(*args, **self.config, **kwargs)
        return self

    def to_triangle(self) -> Triangle:
        pass


def add_node(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        node = func(self, *args, **kwargs)
        if node.name in self.graph:
            raise ValueError(f"Node with name {node.name} already exists.")
        self.graph.append(node)
        return node
    return wrapper


class BaseBermudaAPI(ABC):

    def __init__(self, asynchronous: bool = False) -> None:
        api_key = os.getenv("BERMUDA_API_KEY")
        if api_key is not None:
            print("Found key")
        else:
            raise ValueError("Environment variable not found")
        self.auth = {"Authorization": f"Api-Key: {api_key}"}
        self.graph = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def __getitem__(self, item: str):
        node = [i for i in self.graph if i.name == item]
        if not node:
            raise ValueError(f"Cannot find node {item}.")
        return node[0]


class BermudaAPI(BaseBermudaAPI):

    @add_node
    def triangle(self, triangle: Triangle, name: str, *args, **kwargs) -> TriangleNode:
        return TriangleNode(name)

    @add_node
    def fit(self, name: str, model: Model, *args, **kwargs) -> FitNode:
        return FitNode(name, model, *args, **kwargs)

    @add_node
    def predict(self, model: str, *args, **kwargs) -> PredictNode:
        model = self[model]
        return PredictNode(model.name + " (predict)", model.function, *args, **kwargs)()

