"""
Linear-chain Conditional Random Field (CRF) for POS tagging.

Implementation based on the formulation by Lafferty, McCallum, and
Pereira (2001). Given observation sequence x = (x_1, ..., x_T) and
label sequence y = (y_1, ..., y_T), the conditional probability is:

    P(y|x) = (1 / Z(x)) * exp( sum_t sum_k  w_k * f_k(y_{t-1}, y_t, x, t) )

where Z(x) is the partition function computed via the forward algorithm,
f_k are feature functions, and w_k are the corresponding weights.

This module implements:
    - Forward-backward algorithm for computing Z(x) and marginals
    - Viterbi algorithm for finding the optimal label sequence
    - Stochastic gradient descent (SGD) for parameter estimation
    - Averaged Perceptron for faster convergence on large label sets

The forward-backward and Viterbi passes use vectorized numpy operations
to replace the inner label-loop with matrix broadcasts, reducing wall
time from O(T * L * L) Python iterations to O(T) numpy calls each
operating on (L, L) arrays.

All computations are done in log space to avoid numerical underflow,
following standard practice for sequence models with large label sets
such as the 218-tag MGNN tagset used in Filipino POS tagging.

References:
    Collins, M. (2002). Discriminative Training Methods for Hidden
        Markov Models: Theory and Experiments with Perceptron
        Algorithms. EMNLP.
    Lafferty, J., McCallum, A., & Pereira, F. (2001).
        Conditional Random Fields: Probabilistic Models for Segmenting
        and Labeling Sequence Data. ICML.
"""

import json
import numpy as np


NEG_INF = -1e10


def logsumexp(a, axis=None):
    """Numerically stable log-sum-exp to prevent overflow."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


class CRF:
    """
    Linear-chain CRF for sequence labeling.

    The model learns weights for arbitrary feature functions that
    examine the current observation, surrounding context, and the
    transition between consecutive labels.

    Supports two training algorithms:
        - SGD on negative log-likelihood (train)
        - Structured Averaged Perceptron (train_ap)
    """

    def __init__(self, labels, feature_fn):
        """
        Initialize the CRF with a label set and feature function.

        The feature function should accept (x, i, y_prev, y) and return
        a dict mapping feature names to values. At position 0, y_prev
        is None.
        """
        self.labels = list(labels)
        self.n_labels = len(labels)
        self.label_to_id = {l: i for i, l in enumerate(labels)}
        self.feature_fn = feature_fn
        self.weights = {}
        self._feat_index = {}
        self._w = None

    def _get_feat_id(self, name):
        if name not in self._feat_index:
            fid = len(self._feat_index)
            self._feat_index[name] = fid
            return fid
        return self._feat_index[name]

    def _build_features(self, X, Y):
        """Scan the training data to build the feature-to-index mapping."""
        for x, y in zip(X, Y):
            T = len(x)
            for t in range(T):
                y_prev = y[t - 1] if t > 0 else None
                feats = self.feature_fn(x, t, y_prev, y[t])
                for name in feats:
                    self._get_feat_id(name)
                if t > 0:
                    for y_id in range(self.n_labels):
                        feats = self.feature_fn(x, t, y[t - 1], self.labels[y_id])
                        for name in feats:
                            self._get_feat_id(name)
        n = len(self._feat_index)
        self._w = np.zeros(n, dtype=np.float64)
        for name, val in self.weights.items():
            if name in self._feat_index:
                self._w[self._feat_index[name]] = val

    def _feat_vector(self, x, t, y_prev_id, y_id):
        """Return sparse feature vector as (feat_id, value) pairs."""
        y_prev = self.labels[y_prev_id] if y_prev_id is not None else None
        y = self.labels[y_id]
        feats = self.feature_fn(x, t, y_prev, y)
        sparse = []
        for name, val in feats.items():
            if name in self._feat_index:
                sparse.append((self._feat_index[name], val))
        return sparse

    def _score(self, sparse_feats):
        """Dot product w . f for a sparse feature vector."""
        s = 0.0
        for fid, val in sparse_feats:
            s += self._w[fid] * val
        return s

    def _sequence_feature_vector(self, x, y_ids):
        """Sum feature vectors over an entire label sequence."""
        n = len(self._w)
        fv = np.zeros(n, dtype=np.float64)
        T = len(x)
        for t in range(T):
            yp = y_ids[t - 1] if t > 0 else None
            feats = self._feat_vector(x, t, yp, y_ids[t])
            for fid, val in feats:
                fv[fid] += val
        return fv

    def _potential_table(self, x):
        """
        Precompute log-potential phi[t, y_prev, y] for each position.

        phi[t, y', y] = sum_k w_k * f_k(y', y, x, t)

        At t=0 the y_prev dimension is broadcast (no previous label).
        Returns array of shape (T, L, L).
        """
        T = len(x)
        L = self.n_labels
        phi = np.full((T, L, L), NEG_INF, dtype=np.float64)

        for y_id in range(L):
            feats = self._feat_vector(x, 0, None, y_id)
            phi[0, :, y_id] = self._score(feats)

        for t in range(1, T):
            for yp in range(L):
                for y_id in range(L):
                    feats = self._feat_vector(x, t, yp, y_id)
                    phi[t, yp, y_id] = self._score(feats)

        return phi

    def _forward(self, phi):
        """
        Forward algorithm (Rabiner, 1989) in log space.

        Vectorized: the inner loop over labels is replaced by a
        numpy broadcast on (L,) + (L, L) arrays per time step.
        """
        T, L = phi.shape[0], self.n_labels
        alpha = np.full((T, L), NEG_INF, dtype=np.float64)
        alpha[0] = phi[0, 0, :]

        for t in range(1, T):
            # alpha[t-1] is (L,), phi[t] is (L, L)
            # broadcast: (L, 1) + (L, L) -> (L, L), then logsumexp over axis 0
            alpha[t] = logsumexp(alpha[t - 1][:, np.newaxis] + phi[t], axis=0)

        return alpha

    def _backward(self, phi):
        """
        Backward algorithm in log space.

        Vectorized: (L, L) + (1, L) broadcast per time step.
        """
        T, L = phi.shape[0], self.n_labels
        beta = np.full((T, L), NEG_INF, dtype=np.float64)
        beta[T - 1] = 0.0

        for t in range(T - 2, -1, -1):
            # phi[t+1] is (L, L), beta[t+1] is (L,)
            # broadcast: (L, L) + (1, L) -> (L, L), then logsumexp over axis 1
            beta[t] = logsumexp(phi[t + 1] + beta[t + 1][np.newaxis, :], axis=1)

        return beta

    def _log_partition(self, alpha):
        """Compute log Z(x) from the final row of the forward table."""
        return logsumexp(alpha[-1])

    def _marginals(self, phi, alpha, beta, log_z):
        """
        Edge marginals P(y_{t-1}, y_t | x) via forward-backward.

        Vectorized: each time step computes an (L, L) matrix of
        marginal probabilities in a single numpy expression.
        """
        T, L = phi.shape[0], self.n_labels
        mu = np.zeros((T, L, L), dtype=np.float64)

        for y in range(L):
            mu[0, :, y] = np.exp(alpha[0, y] + beta[0, y] - log_z)

        for t in range(1, T):
            # (L, 1) + (L, L) + (1, L) - scalar -> (L, L)
            mu[t] = np.exp(
                alpha[t - 1][:, np.newaxis] + phi[t] + beta[t][np.newaxis, :] - log_z
            )

        return mu

    def _gradient(self, x, y, phi, mu):
        """
        Gradient of log P(y|x) with respect to w.

        Following Lafferty et al. (2001), the gradient is the difference
        between the observed feature counts and the expected feature
        counts under the model:

            dL/dw_k = sum_t f_k(observed) - sum_t E_p[f_k | x]
        """
        T = len(x)
        L = self.n_labels
        grad = np.zeros_like(self._w)

        y_ids = [self.label_to_id[label] for label in y]

        for t in range(T):
            yp = y_ids[t - 1] if t > 0 else None
            feats = self._feat_vector(x, t, yp, y_ids[t])
            for fid, val in feats:
                grad[fid] += val

        for t in range(T):
            if t == 0:
                for y_id in range(L):
                    p = mu[0, 0, y_id]
                    if p < 1e-15:
                        continue
                    feats = self._feat_vector(x, 0, None, y_id)
                    for fid, val in feats:
                        grad[fid] -= p * val
            else:
                for yp in range(L):
                    for y_id in range(L):
                        p = mu[t, yp, y_id]
                        if p < 1e-15:
                            continue
                        feats = self._feat_vector(x, t, yp, y_id)
                        for fid, val in feats:
                            grad[fid] -= p * val

        return grad

    def train(self, X, Y, max_iter=30, learning_rate=0.01, l2_reg=0.1,
              verbose=True):
        """
        Estimate parameters using SGD on negative log-likelihood.

        L2 regularization (lambda * ||w||^2 / 2) is applied to prevent
        overfitting, following the approach described in Sutton and
        McCallum (2012), "An Introduction to Conditional Random Fields."
        """
        self._build_features(X, Y)
        n_feats = len(self._feat_index)
        if verbose:
            print("Features: %d, Labels: %d, Sequences: %d"
                  % (n_feats, self.n_labels, len(X)))

        for epoch in range(max_iter):
            indices = np.random.permutation(len(X))
            total_loss = 0.0

            for idx in indices:
                x, y = X[idx], Y[idx]
                phi = self._potential_table(x)
                alpha = self._forward(phi)
                beta = self._backward(phi)
                log_z = self._log_partition(alpha)
                mu = self._marginals(phi, alpha, beta, log_z)

                score = 0.0
                y_ids = [self.label_to_id[label] for label in y]
                for t in range(len(x)):
                    yp = y_ids[t - 1] if t > 0 else None
                    feats = self._feat_vector(x, t, yp, y_ids[t])
                    score += self._score(feats)
                total_loss += -(score - log_z)

                grad = self._gradient(x, y, phi, mu)
                self._w += learning_rate * grad
                self._w -= learning_rate * l2_reg * self._w

            total_loss += 0.5 * l2_reg * np.dot(self._w, self._w)

            if verbose:
                print("Epoch %d/%d  loss=%.4f" % (epoch + 1, max_iter, total_loss))

        self._sync_weights_to_dict()

    def train_ap(self, X, Y, max_iter=30, verbose=True):
        """
        Estimate parameters using the Structured Averaged Perceptron.

        The perceptron updates are simpler than SGD: for each training
        sequence, decode with Viterbi; if the prediction differs from
        the gold standard, add the gold feature vector and subtract
        the predicted feature vector. Weights are averaged over all
        updates to improve generalization (Collins, 2002).

        This avoids forward-backward computation entirely during
        training, making it significantly faster than SGD for large
        label sets like the 218-tag MGNN tagset.
        """
        self._build_features(X, Y)
        n_feats = len(self._feat_index)
        if verbose:
            print("Features: %d, Labels: %d, Sequences: %d"
                  % (n_feats, self.n_labels, len(X)))

        w_sum = np.zeros_like(self._w)
        n_updates = 0

        for epoch in range(max_iter):
            indices = np.random.permutation(len(X))
            n_correct = 0
            n_total = 0

            for idx in indices:
                x, y = X[idx], Y[idx]
                y_ids = [self.label_to_id[label] for label in y]

                pred = self.predict(x)
                pred_ids = [self.label_to_id[label] for label in pred]

                if pred != y:
                    gold_fv = self._sequence_feature_vector(x, y_ids)
                    pred_fv = self._sequence_feature_vector(x, pred_ids)
                    self._w += gold_fv - pred_fv

                w_sum += self._w
                n_updates += 1

                for p, g in zip(pred, y):
                    if p == g:
                        n_correct += 1
                    n_total += 1

            acc = n_correct / n_total if n_total > 0 else 0.0
            if verbose:
                print("Epoch %d/%d  acc=%.4f" % (epoch + 1, max_iter, acc))

        self._w = w_sum / n_updates
        self._sync_weights_to_dict()

    def predict(self, x):
        """
        Find the most probable label sequence using Viterbi decoding.

        Vectorized: the inner loop over labels is replaced by numpy
        argmax and broadcasting on (L,) + (L, L) arrays per step.
        """
        phi = self._potential_table(x)
        T, L = len(x), self.n_labels

        delta = np.full((T, L), NEG_INF, dtype=np.float64)
        psi = np.zeros((T, L), dtype=np.int32)

        delta[0] = phi[0, 0, :]

        for t in range(1, T):
            # (L, 1) + (L, L) -> (L, L), then max over axis 0
            scores = delta[t - 1][:, np.newaxis] + phi[t]
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(L)]

        path = [0] * T
        path[T - 1] = int(np.argmax(delta[T - 1]))
        for t in range(T - 2, -1, -1):
            path[t] = int(psi[t + 1, path[t + 1]])

        return [self.labels[i] for i in path]

    def evaluate(self, X, Y):
        """Compute token-level accuracy over a test set."""
        correct = 0
        total = 0
        for x, y in zip(X, Y):
            pred = self.predict(x)
            for p, a in zip(pred, y):
                if p == a:
                    correct += 1
                total += 1
        accuracy = correct / total if total > 0 else 0.0
        return correct, total, accuracy

    def _sync_weights_to_dict(self):
        id_to_name = {v: k for k, v in self._feat_index.items()}
        self.weights = {}
        for fid in range(len(self._w)):
            if abs(self._w[fid]) > 1e-12:
                self.weights[id_to_name[fid]] = float(self._w[fid])

    def save(self, path):
        """Save learned weights and label set to a JSON file."""
        self._sync_weights_to_dict()
        data = {
            "labels": self.labels,
            "feature_index": self._feat_index,
            "weights": self.weights,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        """Load a previously saved model from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.labels = data["labels"]
        self.n_labels = len(self.labels)
        self.label_to_id = {l: i for i, l in enumerate(self.labels)}
        self._feat_index = data["feature_index"]
        self.weights = data["weights"]
        self._w = np.zeros(len(self._feat_index), dtype=np.float64)
        for name, val in self.weights.items():
            if name in self._feat_index:
                self._w[self._feat_index[name]] = val


def pos_features(x, i, y_prev, y):
    """
    Feature function for Filipino POS tagging.

    Extracts the following feature types used in the CRFPOST model:
        - Current word identity and affixes (2- and 3-character)
        - Word shape (capitalization, digit)
        - Previous and next word context (window size = 1)
        - Label transition bigram (y_{t-1} -> y_t)
        - Boundary markers (BOS/EOS)

    These features are designed for the MGNN tagset (218 tags)
    used in the OHTree Filipino treebank.
    """
    word = x[i]
    feats = {}

    feats["bias:%s" % y] = 1
    feats["word=%s:%s" % (word, y)] = 1
    feats["suf3=%s:%s" % (word[-3:], y)] = 1
    feats["suf2=%s:%s" % (word[-2:], y)] = 1
    feats["pre3=%s:%s" % (word[:3], y)] = 1
    feats["pre2=%s:%s" % (word[:2], y)] = 1

    if word.isupper():
        feats["upper:%s" % y] = 1
    if word.istitle():
        feats["title:%s" % y] = 1
    if word.isdigit():
        feats["digit:%s" % y] = 1

    if i > 0:
        w1 = x[i - 1]
        feats["-1:word=%s:%s" % (w1, y)] = 1
        feats["-1:suf3=%s:%s" % (w1[-3:], y)] = 1
    else:
        feats["BOS:%s" % y] = 1

    if i < len(x) - 1:
        w1 = x[i + 1]
        feats["+1:word=%s:%s" % (w1, y)] = 1
        feats["+1:suf3=%s:%s" % (w1[-3:], y)] = 1
    else:
        feats["EOS:%s" % y] = 1

    if y_prev is not None:
        feats["trans=%s|%s" % (y_prev, y)] = 1

    return feats
