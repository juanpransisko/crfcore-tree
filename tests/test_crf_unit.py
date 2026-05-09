"""Unit tests for the linear-chain CRF implementation."""

import os
import json
import tempfile
import numpy as np
import pytest

from crf import CRF, pos_features, logsumexp, NEG_INF


class TestLogsumexp:
    def test_single_value(self):
        assert logsumexp(np.array([3.0])) == pytest.approx(3.0)

    def test_two_equal_values(self):
        result = logsumexp(np.array([1.0, 1.0]))
        assert result == pytest.approx(1.0 + np.log(2.0))

    def test_large_values_no_overflow(self):
        result = logsumexp(np.array([1000.0, 1001.0]))
        assert np.isfinite(result)

    def test_axis(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = logsumexp(a, axis=1)
        assert result.shape == (2,)


class TestPosFeatures:
    def test_returns_dict(self):
        feats = pos_features(['the', 'dog'], 0, None, 'DT')
        assert isinstance(feats, dict)
        assert all(isinstance(v, (int, float)) for v in feats.values())

    def test_bos_feature(self):
        feats = pos_features(['the', 'dog'], 0, None, 'DT')
        assert 'BOS:DT' in feats

    def test_eos_feature(self):
        feats = pos_features(['the', 'dog'], 1, 'DT', 'NN')
        assert 'EOS:NN' in feats

    def test_transition_feature(self):
        feats = pos_features(['the', 'dog'], 1, 'DT', 'NN')
        assert 'trans=DT|NN' in feats

    def test_no_transition_at_start(self):
        feats = pos_features(['the', 'dog'], 0, None, 'DT')
        assert not any(k.startswith('trans=') for k in feats)

    def test_word_feature(self):
        feats = pos_features(['the', 'dog'], 0, None, 'DT')
        assert 'word=the:DT' in feats

    def test_context_features(self):
        feats = pos_features(['the', 'big', 'dog'], 1, 'DT', 'JJ')
        assert any(k.startswith('-1:') for k in feats)
        assert any(k.startswith('+1:') for k in feats)


class TestCRFInit:
    def test_label_mapping(self, toy_labels):
        crf = CRF(labels=toy_labels, feature_fn=pos_features)
        assert crf.n_labels == 3
        assert crf.label_to_id['DT'] == 0
        assert crf.labels[crf.label_to_id['NN']] == 'NN'

    def test_empty_weights(self, toy_labels):
        crf = CRF(labels=toy_labels, feature_fn=pos_features)
        assert crf.weights == {}


class TestCRFForwardBackward:
    def test_forward_shape(self, trained_crf):
        phi = trained_crf._potential_table(['the', 'dog'])
        alpha = trained_crf._forward(phi)
        assert alpha.shape == (2, trained_crf.n_labels)

    def test_backward_shape(self, trained_crf):
        phi = trained_crf._potential_table(['the', 'dog'])
        beta = trained_crf._backward(phi)
        assert beta.shape == (2, trained_crf.n_labels)

    def test_partition_consistency(self, trained_crf):
        phi = trained_crf._potential_table(['the', 'dog'])
        alpha = trained_crf._forward(phi)
        beta = trained_crf._backward(phi)
        log_z_fwd = logsumexp(alpha[-1])
        log_z_bwd = logsumexp(phi[0, 0, :] + beta[0])
        assert log_z_fwd == pytest.approx(log_z_bwd, abs=1e-6)

    def test_marginals_sum_to_one(self, trained_crf):
        phi = trained_crf._potential_table(['the', 'dog', 'runs'])
        alpha = trained_crf._forward(phi)
        beta = trained_crf._backward(phi)
        log_z = trained_crf._log_partition(alpha)
        mu = trained_crf._marginals(phi, alpha, beta, log_z)
        for t in range(1, 3):
            total = mu[t].sum()
            assert total == pytest.approx(1.0, abs=1e-6)


class TestCRFTrain:
    def test_loss_decreases(self, toy_labels, toy_data):
        from crf import CRF, pos_features
        X, Y = toy_data
        crf = CRF(labels=toy_labels, feature_fn=pos_features)
        crf._build_features(X, Y)
        crf._w = np.zeros(len(crf._feat_index))

        losses = []
        for epoch in range(5):
            total_loss = 0.0
            for x, y in zip(X, Y):
                phi = crf._potential_table(x)
                alpha = crf._forward(phi)
                log_z = crf._log_partition(alpha)
                beta = crf._backward(phi)
                mu = crf._marginals(phi, alpha, beta, log_z)
                score = 0.0
                y_ids = [crf.label_to_id[l] for l in y]
                for t in range(len(x)):
                    yp = y_ids[t - 1] if t > 0 else None
                    feats = crf._feat_vector(x, t, yp, y_ids[t])
                    score += crf._score(feats)
                total_loss += -(score - log_z)
                grad = crf._gradient(x, y, phi, mu)
                crf._w += 0.1 * grad
            losses.append(total_loss)

        assert losses[-1] < losses[0]

    def test_weights_nonzero_after_training(self, trained_crf):
        assert any(abs(w) > 1e-10 for w in trained_crf._w)

    def test_feature_index_populated(self, trained_crf):
        assert len(trained_crf._feat_index) > 0


class TestCRFPredict:
    def test_predict_returns_labels(self, trained_crf, toy_labels):
        pred = trained_crf.predict(['the', 'cat', 'runs'])
        assert len(pred) == 3
        assert all(p in toy_labels for p in pred)

    def test_predict_correct_on_training_data(self, trained_crf, toy_data):
        X, Y = toy_data
        for x, y in zip(X, Y):
            pred = trained_crf.predict(x)
            assert pred == y

    def test_predict_single_token(self, trained_crf, toy_labels):
        pred = trained_crf.predict(['dog'])
        assert len(pred) == 1
        assert pred[0] in toy_labels


class TestCRFEvaluate:
    def test_evaluate_perfect(self, trained_crf, toy_data):
        X, Y = toy_data
        correct, total, acc = trained_crf.evaluate(X, Y)
        assert acc == 1.0
        assert correct == total

    def test_evaluate_counts(self, trained_crf):
        X = [['x', 'y']]
        Y = [['DT', 'DT']]
        correct, total, acc = trained_crf.evaluate(X, Y)
        assert total == 2


class TestCRFSaveLoad:
    def test_roundtrip(self, trained_crf, toy_labels):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            trained_crf.save(path)
            loaded = CRF(labels=[], feature_fn=pos_features)
            loaded.load(path)

            assert loaded.labels == toy_labels
            assert loaded.n_labels == len(toy_labels)

            pred_orig = trained_crf.predict(['the', 'dog', 'runs'])
            pred_loaded = loaded.predict(['the', 'dog', 'runs'])
            assert pred_orig == pred_loaded
        finally:
            os.remove(path)

    def test_save_creates_file(self, trained_crf):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            trained_crf.save(path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert 'labels' in data
            assert 'weights' in data
            assert 'feature_index' in data
        finally:
            os.remove(path)

    def test_save_only_nonzero_weights(self, trained_crf):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            trained_crf.save(path)
            with open(path) as f:
                data = json.load(f)
            for w in data['weights'].values():
                assert abs(w) > 1e-12
        finally:
            os.remove(path)
