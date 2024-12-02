import numpy as np
from abc import ABC, abstractmethod
from typing import Dict


class AnnotatorModel(ABC):
    """Abstract base class for coding models"""

    def __init__(self, n_classes: int, n_coders: int):
        self.K = n_classes
        self.J = n_coders
        self.alpha = np.ones(self.K) / self.K  # Class priors

    @abstractmethod
    def calc_logliks(self, data: Dict) -> Dict:
        """Calculate log-likelihoods and posterior probabilities"""
        pass

    @abstractmethod
    def map_update(self, data: Dict, prior: float = 1.0) -> Dict:
        """Update parameters using EM"""
        pass
