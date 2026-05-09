"""Functional tests for the CRFPOST training and evaluation pipeline."""

import os
import tempfile
import shutil
import pytest

from crf import CRF, pos_features


@pytest.fixture
def pipeline_dir():
    """Create a temporary directory mimicking the crfcore-tree layout."""
    d = tempfile.mkdtemp(prefix='crftest_')
    os.makedirs(os.path.join(d, 'corpus'))
    os.makedirs(os.path.join(d, 'models'))
    os.makedirs(os.path.join(d, 'results'))

    train_words = [
        'the dog runs',
        'a cat sleeps',
        'the bird flies',
        'a fish swims',
        'the boy walks',
    ]
    train_tags = [
        'DT NN VB',
        'DT NN VB',
        'DT NN VB',
        'DT NN VB',
        'DT NN VB',
    ]
    test_words = [
        'the cat runs',
        'a dog flies',
    ]
    test_tags = [
        'DT NN VB',
        'DT NN VB',
    ]

    for name, lines in [
        ('trainHPOST.words', train_words),
        ('trainHPOST.tags', train_tags),
        ('testHPOST.words', test_words),
        ('testHPOST.tags', test_tags),
    ]:
        with open(os.path.join(d, 'corpus', name), 'w') as f:
            f.write('\n'.join(lines) + '\n')

    yield d
    shutil.rmtree(d)


class TestTrainEvalPipeline:
    def test_train_produces_model(self, pipeline_dir):
        corpus_dir = os.path.join(pipeline_dir, 'corpus')
        model_path = os.path.join(pipeline_dir, 'models', 'test_model.json')

        X, Y = _load_corpus(corpus_dir, 'train')
        labels = sorted(set(tag for seq in Y for tag in seq))

        crf = CRF(labels=labels, feature_fn=pos_features)
        crf.train(X, Y, max_iter=10, learning_rate=0.1, l2_reg=0.01, verbose=False)
        crf.save(model_path)

        assert os.path.exists(model_path)
        assert os.path.getsize(model_path) > 0

    def test_eval_after_train(self, pipeline_dir):
        corpus_dir = os.path.join(pipeline_dir, 'corpus')
        model_path = os.path.join(pipeline_dir, 'models', 'test_model.json')

        X_train, Y_train = _load_corpus(corpus_dir, 'train')
        X_test, Y_test = _load_corpus(corpus_dir, 'test')
        labels = sorted(set(tag for seq in Y_train for tag in seq))

        crf = CRF(labels=labels, feature_fn=pos_features)
        crf.train(X_train, Y_train, max_iter=10, learning_rate=0.1,
                  l2_reg=0.01, verbose=False)
        crf.save(model_path)

        loaded = CRF(labels=[], feature_fn=pos_features)
        loaded.load(model_path)

        correct, total, acc = loaded.evaluate(X_test, Y_test)
        assert total == 6
        assert acc > 0.5

    def test_predictions_match_sequence_length(self, pipeline_dir):
        corpus_dir = os.path.join(pipeline_dir, 'corpus')
        X_train, Y_train = _load_corpus(corpus_dir, 'train')
        labels = sorted(set(tag for seq in Y_train for tag in seq))

        crf = CRF(labels=labels, feature_fn=pos_features)
        crf.train(X_train, Y_train, max_iter=5, learning_rate=0.1,
                  l2_reg=0.01, verbose=False)

        for x in X_train:
            pred = crf.predict(x)
            assert len(pred) == len(x)

    def test_train_on_toy_reaches_high_accuracy(self, pipeline_dir):
        corpus_dir = os.path.join(pipeline_dir, 'corpus')
        X, Y = _load_corpus(corpus_dir, 'train')
        labels = sorted(set(tag for seq in Y for tag in seq))

        crf = CRF(labels=labels, feature_fn=pos_features)
        crf.train(X, Y, max_iter=20, learning_rate=0.1, l2_reg=0.01, verbose=False)

        _, _, acc = crf.evaluate(X, Y)
        assert acc == 1.0


class TestEvalOutputs:
    def test_conll_format(self, pipeline_dir):
        corpus_dir = os.path.join(pipeline_dir, 'corpus')
        results_dir = os.path.join(pipeline_dir, 'results')

        X_train, Y_train = _load_corpus(corpus_dir, 'train')
        X_test, Y_test = _load_corpus(corpus_dir, 'test')
        labels = sorted(set(tag for seq in Y_train for tag in seq))

        crf = CRF(labels=labels, feature_fn=pos_features)
        crf.train(X_train, Y_train, max_iter=10, learning_rate=0.1,
                  l2_reg=0.01, verbose=False)

        conll_path = os.path.join(results_dir, 'test_predictions.conll')
        with open(conll_path, 'w') as f:
            for x, gold in zip(X_test, Y_test):
                pred = crf.predict(x)
                for w, p in zip(x, pred):
                    f.write('%s\t%s\n' % (w, p))
                f.write('\n')

        with open(conll_path) as f:
            lines = f.readlines()

        non_blank = [l for l in lines if l.strip()]
        for line in non_blank:
            parts = line.strip().split('\t')
            assert len(parts) == 2

    def test_comparison_format(self, pipeline_dir):
        corpus_dir = os.path.join(pipeline_dir, 'corpus')
        results_dir = os.path.join(pipeline_dir, 'results')

        X_train, Y_train = _load_corpus(corpus_dir, 'train')
        X_test, Y_test = _load_corpus(corpus_dir, 'test')
        labels = sorted(set(tag for seq in Y_train for tag in seq))

        crf = CRF(labels=labels, feature_fn=pos_features)
        crf.train(X_train, Y_train, max_iter=10, learning_rate=0.1,
                  l2_reg=0.01, verbose=False)

        compare_path = os.path.join(results_dir, 'test_comparison.txt')
        with open(compare_path, 'w') as f:
            f.write('%-20s %-12s %-12s %s\n' % ('WORD', 'GOLD', 'PREDICTED', 'MATCH'))
            f.write('-' * 60 + '\n')
            for x, gold in zip(X_test, Y_test):
                pred = crf.predict(x)
                for w, g, p in zip(x, gold, pred):
                    match = '.' if g == p else 'X'
                    f.write('%-20s %-12s %-12s %s\n' % (w, g, p, match))
                f.write('\n')

        with open(compare_path) as f:
            content = f.read()

        assert 'WORD' in content
        assert 'GOLD' in content
        assert 'PREDICTED' in content


class TestEdgeCases:
    def test_single_token_sequence(self):
        labels = ['NN', 'VB']
        X = [['dog'], ['run']]
        Y = [['NN'], ['VB']]

        crf = CRF(labels=labels, feature_fn=pos_features)
        crf.train(X, Y, max_iter=10, learning_rate=0.1, l2_reg=0.01, verbose=False)

        pred = crf.predict(['dog'])
        assert len(pred) == 1
        assert pred[0] in labels

    def test_unseen_word(self):
        labels = ['DT', 'NN', 'VB']
        X = [['the', 'dog', 'runs']]
        Y = [['DT', 'NN', 'VB']]

        crf = CRF(labels=labels, feature_fn=pos_features)
        crf.train(X, Y, max_iter=10, learning_rate=0.1, l2_reg=0.01, verbose=False)

        pred = crf.predict(['a', 'elephant', 'dances'])
        assert len(pred) == 3
        assert all(p in labels for p in pred)

    def test_long_sequence(self):
        labels = ['A', 'B']
        X = [['w%d' % i for i in range(50)]]
        Y = [['A' if i % 2 == 0 else 'B' for i in range(50)]]

        crf = CRF(labels=labels, feature_fn=pos_features)
        crf.train(X, Y, max_iter=5, learning_rate=0.1, l2_reg=0.01, verbose=False)

        pred = crf.predict(X[0])
        assert len(pred) == 50


def _load_corpus(corpus_dir, split):
    words_file = os.path.join(corpus_dir, '%sHPOST.words' % split)
    tags_file = os.path.join(corpus_dir, '%sHPOST.tags' % split)
    with open(words_file) as wf, open(tags_file) as tf:
        words_lines = wf.readlines()
        tags_lines = tf.readlines()
    X, Y = [], []
    for w, t in zip(words_lines, tags_lines):
        ws = w.strip().split()
        ts = t.strip().split()
        if ws and ts and len(ws) == len(ts):
            X.append(ws)
            Y.append(ts)
    return X, Y
