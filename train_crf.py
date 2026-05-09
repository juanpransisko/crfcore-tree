"""
Training script for the CRFPOST model using python-crfsuite.

Trains a linear-chain CRF on the OHTree Filipino treebank corpus
annotated with the MGNN tagset (218 POS tags). The implementation
uses the Averaged Perceptron (AP) algorithm, which offers faster
convergence than L-BFGS for large label sets (Collins, 2002).

Feature extraction follows the template used in the CRFPOST system:
word identity, character n-gram affixes (2- and 3-character prefixes
and suffixes), orthographic properties (capitalization, digit), and
a context window of size 1 (previous and next word).

After training, the script evaluates on a held-out test set and
writes predictions in CoNLL format, a side-by-side comparison of
gold vs. predicted tags, and a summary of accuracy metrics.

Usage:
    python train_crf.py

References:
    Collins, M. (2002). Discriminative Training Methods for Hidden
        Markov Models: Theory and Experiments with Perceptron
        Algorithms. EMNLP.
"""

import os
import time
import pycrfsuite


def word_features(sent, i):
    word = sent[i]
    features = [
        'bias',
        'word=' + word,
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word[:3]=' + word[:3],
        'word[:2]=' + word[:2],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
    ]
    if i > 0:
        word1 = sent[i-1]
        features.extend([
            '-1:word=' + word1,
            '-1:word[-3:]=' + word1[-3:],
            '-1:word[-2:]=' + word1[-2:],
            '-1:word.istitle=%s' % word1.istitle(),
        ])
    else:
        features.append('BOS')
    if i < len(sent)-1:
        word1 = sent[i+1]
        features.extend([
            '+1:word=' + word1,
            '+1:word[-3:]=' + word1[-3:],
            '+1:word[-2:]=' + word1[-2:],
            '+1:word.istitle=%s' % word1.istitle(),
        ])
    else:
        features.append('EOS')
    return features


def sent_features(sent):
    return [word_features(sent, i) for i in range(len(sent))]


def load_raw(words_file, tags_file):
    with open(words_file) as wf, open(tags_file) as tf:
        words_lines = wf.readlines()
        tags_lines = tf.readlines()
    words, tags = [], []
    for w, t in zip(words_lines, tags_lines):
        ws = w.strip().split()
        ts = t.strip().split()
        if ws and ts:
            words.append(ws)
            tags.append(ts)
    return words, tags


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root, 'models', 'crfpost_model.crfsuite')
    results_dir = os.path.join(root, 'results')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_words = os.path.join(root, 'corpus', 'trainHPOST.words')
    train_tags = os.path.join(root, 'corpus', 'trainHPOST.tags')
    test_words = os.path.join(root, 'corpus', 'testHPOST.words')
    test_tags = os.path.join(root, 'corpus', 'testHPOST.tags')

    print('Loading training data...')
    train_w, train_t = load_raw(train_words, train_tags)
    print('Loaded %d training sentences' % len(train_w))

    print('Training CRF model...')
    trainer = pycrfsuite.Trainer(verbose=True)
    for ws, ts in zip(train_w, train_t):
        trainer.append(sent_features(ws), ts)

    trainer.select('ap')
    trainer.set_params({
        'max_iterations': 30,
        'feature.possible_transitions': False,
        'feature.minfreq': 3,
    })

    start = time.time()
    trainer.train(model_path)
    elapsed = time.time() - start
    print('Training completed in %.1fs' % elapsed)
    print('Model saved to: %s' % model_path)

    print('\nEvaluating on test data...')
    test_w, test_t = load_raw(test_words, test_tags)
    print('Loaded %d test sentences' % len(test_w))

    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)

    correct = 0
    total = 0
    all_predictions = []

    for ws, gold in zip(test_w, test_t):
        pred = tagger.tag(sent_features(ws))
        all_predictions.append((ws, gold, pred))
        for p, g in zip(pred, gold):
            if p == g:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print('Accuracy: %d/%d = %.2f%%' % (correct, total, accuracy))
    print('Labels: %d' % len(tagger.labels()))

    conll_path = os.path.join(results_dir, 'crfsuite_predictions.conll')
    with open(conll_path, 'w', encoding='utf-8') as f:
        for ws, gold, pred in all_predictions:
            for w, p in zip(ws, pred):
                f.write('%s\t%s\n' % (w, p))
            f.write('\n')
    print('Predictions saved to: %s' % conll_path)

    compare_path = os.path.join(results_dir, 'crfsuite_comparison.txt')
    with open(compare_path, 'w', encoding='utf-8') as f:
        f.write('%-20s %-12s %-12s %s\n' % ('WORD', 'GOLD', 'PREDICTED', 'MATCH'))
        f.write('-' * 60 + '\n')
        for ws, gold, pred in all_predictions:
            for w, g, p in zip(ws, gold, pred):
                match = '.' if g == p else 'X'
                f.write('%-20s %-12s %-12s %s\n' % (w, g, p, match))
            f.write('\n')
    print('Comparison saved to: %s' % compare_path)

    metrics_path = os.path.join(results_dir, 'crfsuite_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write('Algorithm: ap\n')
        f.write('Training sentences: %d\n' % len(train_w))
        f.write('Test sentences: %d\n' % len(test_w))
        f.write('Training time: %.1fs\n' % elapsed)
        f.write('Labels: %d\n' % len(tagger.labels()))
        f.write('Correct: %d\n' % correct)
        f.write('Total: %d\n' % total)
        f.write('Accuracy: %.2f%%\n' % accuracy)
        f.write('Model: %s\n' % model_path)
    print('Metrics saved to: %s' % metrics_path)


if __name__ == '__main__':
    main()
