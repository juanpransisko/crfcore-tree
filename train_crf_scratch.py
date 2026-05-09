"""
Training script for the from-scratch CRFPOST model.

Trains a linear-chain CRF on the OHTree Filipino treebank corpus
using stochastic gradient descent (SGD) with L2 regularization,
as described in Sutton and McCallum (2012). The CRF implementation
in crf.py handles forward-backward inference and gradient
computation; this script manages the data pipeline and evaluation.

Parameter estimation follows the formulation by Lafferty, McCallum,
and Pereira (2001), where the gradient of the log-likelihood is the
difference between observed and expected feature counts under the
model distribution.

After training, the script evaluates on a held-out test set and
writes predictions in CoNLL format, a side-by-side comparison of
gold vs. predicted tags, and a summary of accuracy metrics.

Usage:
    python train_crf_scratch.py

References:
    Lafferty, J., McCallum, A., & Pereira, F. (2001).
        Conditional Random Fields: Probabilistic Models for Segmenting
        and Labeling Sequence Data. ICML.
    Sutton, C. & McCallum, A. (2012). An Introduction to Conditional
        Random Fields. Foundations and Trends in Machine Learning.
"""

import os
import time

from crf import CRF, pos_features


def load_data(words_file, tags_file):
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


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root, 'models', 'crf_scratch_model.json')
    results_dir = os.path.join(root, 'results')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_words = os.path.join(root, 'corpus', 'trainHPOST.words')
    train_tags = os.path.join(root, 'corpus', 'trainHPOST.tags')
    test_words = os.path.join(root, 'corpus', 'testHPOST.words')
    test_tags = os.path.join(root, 'corpus', 'testHPOST.tags')

    print('Loading training data...')
    X_train, Y_train = load_data(train_words, train_tags)
    print('Loaded %d training sentences' % len(X_train))

    print('Loading test data...')
    X_test, Y_test = load_data(test_words, test_tags)
    print('Loaded %d test sentences' % len(X_test))

    labels = sorted(set(tag for seq in Y_train for tag in seq))
    print('Labels: %d' % len(labels))

    crf = CRF(labels=labels, feature_fn=pos_features)

    print('\nTraining...')
    start = time.time()
    crf.train(X_train, Y_train, max_iter=30, learning_rate=0.01,
              l2_reg=0.1, verbose=True)
    elapsed = time.time() - start
    print('Training completed in %.1fs' % elapsed)

    crf.save(model_path)
    print('Model saved to: %s' % model_path)

    print('\nEvaluating on test data...')
    correct = 0
    total = 0
    all_predictions = []

    for x, gold in zip(X_test, Y_test):
        pred = crf.predict(x)
        all_predictions.append((x, gold, pred))
        for p, g in zip(pred, gold):
            if p == g:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print('Accuracy: %d/%d = %.2f%%' % (correct, total, accuracy * 100))

    conll_path = os.path.join(results_dir, 'scratch_predictions.conll')
    with open(conll_path, 'w', encoding='utf-8') as f:
        for ws, gold, pred in all_predictions:
            for w, p in zip(ws, pred):
                f.write('%s\t%s\n' % (w, p))
            f.write('\n')
    print('Predictions saved to: %s' % conll_path)

    compare_path = os.path.join(results_dir, 'scratch_comparison.txt')
    with open(compare_path, 'w', encoding='utf-8') as f:
        f.write('%-20s %-12s %-12s %s\n' % ('WORD', 'GOLD', 'PREDICTED', 'MATCH'))
        f.write('-' * 60 + '\n')
        for ws, gold, pred in all_predictions:
            for w, g, p in zip(ws, gold, pred):
                match = '.' if g == p else 'X'
                f.write('%-20s %-12s %-12s %s\n' % (w, g, p, match))
            f.write('\n')
    print('Comparison saved to: %s' % compare_path)

    metrics_path = os.path.join(results_dir, 'scratch_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write('Algorithm: SGD (from-scratch CRF)\n')
        f.write('Training sentences: %d\n' % len(X_train))
        f.write('Test sentences: %d\n' % len(X_test))
        f.write('Training time: %.1fs\n' % elapsed)
        f.write('Labels: %d\n' % len(labels))
        f.write('Correct: %d\n' % correct)
        f.write('Total: %d\n' % total)
        f.write('Accuracy: %.2f%%\n' % (accuracy * 100))
        f.write('Model: %s\n' % model_path)
    print('Metrics saved to: %s' % metrics_path)


if __name__ == '__main__':
    main()
