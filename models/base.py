from abc import ABC, abstractmethod


class ModelBuilder(ABC):
    @abstractmethod
    def getModel(self):
        pass