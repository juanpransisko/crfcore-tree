import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def toy_labels():
    return ['DT', 'NN', 'VB']


@pytest.fixture
def toy_data():
    X = [
        ['the', 'dog', 'runs'],
        ['a', 'cat', 'sleeps'],
        ['the', 'bird', 'flies'],
    ]
    Y = [
        ['DT', 'NN', 'VB'],
        ['DT', 'NN', 'VB'],
        ['DT', 'NN', 'VB'],
    ]
    return X, Y


@pytest.fixture
def trained_crf(toy_labels, toy_data):
    from crf import CRF, pos_features
    X, Y = toy_data
    crf = CRF(labels=toy_labels, feature_fn=pos_features)
    crf.train(X, Y, max_iter=10, learning_rate=0.1, l2_reg=0.01, verbose=False)
    return crf
