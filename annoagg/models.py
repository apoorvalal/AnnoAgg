import numpy as np
from scipy.special import logsumexp, softmax

from typing import Dict

from .base import AnnotatorModel


class MACEModel(AnnotatorModel):
    """Multi-Annotator Competence Estimation"""

    def __init__(self, n_classes: int, n_coders: int):
        super().__init__(n_classes, n_coders)
        self.beta = np.full(self.J, 0.8)  # Coder reliabilities

    def calc_logliks(self, data: Dict) -> Dict:
        N, I = len(data["ii"]), data["ii"].max()

        # p(x|z) = β when correct + (1-β)/K when guessing
        ll_dict = np.zeros((self.J, self.K, self.K))
        for j in range(self.J):
            for k in range(self.K):
                correct_prob = self.beta[j] + (1 - self.beta[j]) / self.K
                incorrect_prob = (1 - self.beta[j]) / self.K
                ll_dict[j, k, k] = np.log(correct_prob)
                ll_dict[j, k, :k] = np.log(incorrect_prob)
                ll_dict[j, k, k + 1 :] = np.log(incorrect_prob)

        lpy = np.tile(np.log(self.alpha), (I, 1))
        for n in range(N):
            i, j, val = data["ii"][n] - 1, data["jj"][n] - 1, data["yy"][n] - 1
            lpy[i] += ll_dict[j, :, val]

        post_Z = np.zeros((I, self.K))
        logliks = np.zeros(I)
        for i in range(I):
            logliks[i] = logsumexp(lpy[i])
            post_Z[i] = softmax(lpy[i])

        return {"logliks": logliks, "post_Z": post_Z, "ll_dict": ll_dict}

    def map_update(self, data: Dict, prior: float = 1.0) -> Dict:
        """Update parameters using EM algorithm"""
        results = self.calc_logliks(data)
        post_Z = results["post_Z"]
        p_dict = np.exp(results["ll_dict"])
        logliks = results["logliks"]

        # Update alpha
        alpha = np.full(self.K, prior)
        alpha += post_Z.sum(axis=0)
        self.alpha = alpha / alpha.sum()

        # Calculate v_ij for beta update
        I = data["ii"].max()
        post_C_1 = np.zeros((I, self.J))
        for n in range(len(data["ii"])):
            i, j, val = data["ii"][n] - 1, data["jj"][n] - 1, data["yy"][n] - 1
            post_C_1[i, j] = self.beta[j] * post_Z[i, val] / p_dict[j, val, val]

        # Update beta
        beta_1 = np.full(self.J, prior)
        beta_0 = np.full(self.J, prior)
        for n in range(len(data["ii"])):
            i, j = data["ii"][n] - 1, data["jj"][n] - 1
            beta_1[j] += post_C_1[i, j]
            beta_0[j] += 1 - post_C_1[i, j]
        self.beta = beta_1 / (beta_1 + beta_0)

        return {"alpha": self.alpha, "beta": self.beta, "mean_LL": np.mean(logliks)}


class BACEModel(AnnotatorModel):
    """Bayesian Aggregation of Coder Expertise"""

    def __init__(self, n_classes: int, n_coders: int):
        super().__init__(n_classes, n_coders)
        self.beta = np.full(self.J, 0.8)  # Coder reliabilities
        self.gamma = (
            np.ones((self.J, self.K)) / self.K
        )  # Coder-specific guessing distributions

    def calc_logliks(self, data: Dict) -> Dict:
        N, I = len(data["ii"]), data["ii"].max()

        # p(x|z) = β when correct + (1-β)γ when guessing
        ll_dict = np.zeros((self.J, self.K, self.K))
        for j in range(self.J):
            for k in range(self.K):
                correct_prob = self.beta[j]
                guess_probs = (1 - self.beta[j]) * self.gamma[j]
                ll_dict[j, k, k] = np.log(correct_prob + guess_probs[k])
                ll_dict[j, k, :k] = np.log(guess_probs[:k])
                ll_dict[j, k, k + 1 :] = np.log(guess_probs[k + 1 :])

        lpy = np.tile(np.log(self.alpha), (I, 1))
        for n in range(N):
            i, j, val = data["ii"][n] - 1, data["jj"][n] - 1, data["yy"][n] - 1
            lpy[i] += ll_dict[j, :, val]

        post_Z = np.zeros((I, self.K))
        logliks = np.zeros(I)
        for i in range(I):
            logliks[i] = logsumexp(lpy[i])
            post_Z[i] = softmax(lpy[i])

        return {"logliks": logliks, "post_Z": post_Z, "ll_dict": ll_dict}

    def map_update(self, data: Dict, prior: float = 1.0) -> Dict:
        """Update parameters using EM algorithm"""
        results = self.calc_logliks(data)
        post_Z = results["post_Z"]
        p_dict = np.exp(results["ll_dict"])
        logliks = results["logliks"]

        # Update alpha
        alpha = np.full(self.K, prior)
        alpha += post_Z.sum(axis=0)
        self.alpha = alpha / alpha.sum()

        # Calculate v_ij
        I = data["ii"].max()
        post_C_1 = np.zeros((I, self.J))
        for n in range(len(data["ii"])):
            i, j, val = data["ii"][n] - 1, data["jj"][n] - 1, data["yy"][n] - 1
            post_C_1[i, j] = self.beta[j] * post_Z[i, val] / p_dict[j, val, val]

        # Update beta
        beta_1 = np.full(self.J, prior)
        beta_0 = np.full(self.J, prior)
        for n in range(len(data["ii"])):
            i, j = data["ii"][n] - 1, data["jj"][n] - 1
            beta_1[j] += post_C_1[i, j]
            beta_0[j] += 1 - post_C_1[i, j]
        self.beta = beta_1 / (beta_1 + beta_0)

        # Update gamma
        gamma = np.full((self.J, self.K), prior)
        for n in range(len(data["ii"])):
            i, j, val = data["ii"][n] - 1, data["jj"][n] - 1, data["yy"][n] - 1
            gamma[j, val] += 1 - post_C_1[i, j]
        for j in range(self.J):
            self.gamma[j] = gamma[j] / gamma[j].sum()

        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "mean_LL": np.mean(logliks),
        }


class DSModel(AnnotatorModel):
    """Dawid-Skene Model"""

    def __init__(self, n_classes: int, n_coders: int):
        super().__init__(n_classes, n_coders)
        # Initialize confusion matrices with high diagonal values
        self.pi = np.zeros((self.J, self.K, self.K))
        for j in range(self.J):
            for k in range(self.K):
                self.pi[j, k, k] = 0.8
                self.pi[j, k, :k] = 0.2 / (self.K - 1)
                self.pi[j, k, k + 1 :] = 0.2 / (self.K - 1)

    def calc_logliks(self, data: Dict) -> Dict:
        N, I = len(data["ii"]), data["ii"].max()

        # Use full confusion matrices
        ll_dict = np.log(self.pi + 1e-10)  # Add small constant for numerical stability

        lpy = np.tile(np.log(self.alpha), (I, 1))
        for n in range(N):
            i, j, val = data["ii"][n] - 1, data["jj"][n] - 1, data["yy"][n] - 1
            lpy[i] += ll_dict[j, :, val]

        post_Z = np.zeros((I, self.K))
        logliks = np.zeros(I)
        for i in range(I):
            logliks[i] = logsumexp(lpy[i])
            post_Z[i] = softmax(lpy[i])

        return {"logliks": logliks, "post_Z": post_Z, "ll_dict": ll_dict}

    def map_update(self, data: Dict, prior: float = 1.0) -> Dict:
        """Update parameters using EM algorithm"""
        results = self.calc_logliks(data)
        post_Z = results["post_Z"]
        logliks = results["logliks"]

        # Update alpha
        alpha = np.full(self.K, prior)
        alpha += post_Z.sum(axis=0)
        self.alpha = alpha / alpha.sum()

        # Update confusion matrices
        pi = np.full((self.J, self.K, self.K), prior)
        for n in range(len(data["ii"])):
            i, j, val = data["ii"][n] - 1, data["jj"][n] - 1, data["yy"][n] - 1
            pi[j, :, val] += post_Z[i, :]

        # Normalize confusion matrices
        for j in range(self.J):
            for k in range(self.K):
                self.pi[j, k] = pi[j, k] / pi[j, k].sum()

        return {"alpha": self.alpha, "pi": self.pi, "mean_LL": np.mean(logliks)}
