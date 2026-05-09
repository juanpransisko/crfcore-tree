"""
Evaluation script for the CRFPOST POS tagger.

Loads a trained model (either python-crfsuite or the from-scratch
linear-chain CRF), runs Viterbi decoding on the held-out test
partition of the OHTree Filipino treebank, and reports:

    - Token-level accuracy
    - Per-tag precision, recall, and F1 (micro and macro averaged)
    - Confusion matrix over the most frequent tags

The per-tag breakdown is important for the MGNN tagset (218 tags)
because macro-averaged F1 is heavily influenced by rare compound
tags with few support instances. Weighted F1 gives a more balanced
picture of overall tagger performance.

Usage:
    python eval_crf.py                    # defaults to crfsuite model
    python eval_crf.py --model crfsuite   # python-crfsuite model
    python eval_crf.py --model scratch    # from-scratch JSON model
"""

import os
import argparse
from collections import defaultdict


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


def load_crfsuite_tagger(model_path):
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

    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)

    def predict(tokens):
        feats = [word_features(tokens, i) for i in range(len(tokens))]
        return tagger.tag(feats)

    return predict


def load_scratch_tagger(model_path):
    from crf import CRF, pos_features

    crf = CRF(labels=[], feature_fn=pos_features)
    crf.load(model_path)
    return crf.predict


def compute_per_tag_metrics(all_predictions):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for ws, gold, pred in all_predictions:
        for g, p in zip(gold, pred):
            if g == p:
                tp[g] += 1
            else:
                fp[p] += 1
                fn[g] += 1

    all_tags = sorted(set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())))
    rows = []
    for tag in all_tags:
        t = tp[tag]
        f_p = fp[tag]
        f_n = fn[tag]
        precision = t / (t + f_p) if (t + f_p) > 0 else 0.0
        recall = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = t + f_n
        rows.append((tag, precision, recall, f1, support))

    return rows


def compute_confusion(all_predictions, top_n=30):
    tag_count = defaultdict(int)
    for ws, gold, pred in all_predictions:
        for g in gold:
            tag_count[g] += 1

    top_tags = [t for t, _ in sorted(tag_count.items(), key=lambda x: -x[1])[:top_n]]
    tag_set = set(top_tags)
    tag_idx = {t: i for i, t in enumerate(top_tags)}

    n = len(top_tags)
    matrix = [[0] * n for _ in range(n)]

    for ws, gold, pred in all_predictions:
        for g, p in zip(gold, pred):
            if g in tag_set and p in tag_set:
                matrix[tag_idx[g]][tag_idx[p]] += 1

    return top_tags, matrix


def write_conll(path, all_predictions):
    with open(path, 'w', encoding='utf-8') as f:
        for ws, gold, pred in all_predictions:
            for w, p in zip(ws, pred):
                f.write('%s\t%s\n' % (w, p))
            f.write('\n')


def write_comparison(path, all_predictions):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('%-20s %-12s %-12s %s\n' % ('WORD', 'GOLD', 'PREDICTED', 'MATCH'))
        f.write('-' * 60 + '\n')
        for ws, gold, pred in all_predictions:
            for w, g, p in zip(ws, gold, pred):
                match = '.' if g == p else 'X'
                f.write('%-20s %-12s %-12s %s\n' % (w, g, p, match))
            f.write('\n')


def write_per_tag(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('%-16s %8s %8s %8s %8s\n' % ('TAG', 'PREC', 'RECALL', 'F1', 'SUPPORT'))
        f.write('-' * 56 + '\n')
        for tag, prec, rec, f1, support in rows:
            f.write('%-16s %8.4f %8.4f %8.4f %8d\n' % (tag, prec, rec, f1, support))

        total_support = sum(r[4] for r in rows)
        macro_p = sum(r[1] for r in rows) / len(rows) if rows else 0
        macro_r = sum(r[2] for r in rows) / len(rows) if rows else 0
        macro_f1 = sum(r[3] for r in rows) / len(rows) if rows else 0
        weighted_f1 = sum(r[3] * r[4] for r in rows) / total_support if total_support else 0

        f.write('-' * 56 + '\n')
        f.write('%-16s %8.4f %8.4f %8.4f %8d\n' % ('macro avg', macro_p, macro_r, macro_f1, total_support))
        f.write('%-16s %8s %8s %8.4f %8d\n' % ('weighted avg', '', '', weighted_f1, total_support))


def write_confusion(path, tags, matrix):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('Confusion matrix (top %d tags by frequency)\n' % len(tags))
        f.write('Rows = gold, Columns = predicted\n\n')

        header = '%-12s' % ''
        for t in tags:
            header += '%6s' % t[:6]
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')

        for i, tag in enumerate(tags):
            row = '%-12s' % tag
            for j in range(len(tags)):
                val = matrix[i][j]
                row += '%6d' % val if val > 0 else '%6s' % '.'
            f.write(row + '\n')


def main():
    parser = argparse.ArgumentParser(description='Evaluate CRF POS tagger')
    parser.add_argument('--model', choices=['crfsuite', 'scratch'], default='crfsuite',
                        help='Model type to evaluate (default: crfsuite)')
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    test_words = os.path.join(root, 'corpus', 'testHPOST.words')
    test_tags = os.path.join(root, 'corpus', 'testHPOST.tags')

    if args.model == 'crfsuite':
        model_path = os.path.join(root, 'models', 'crfpost_model.crfsuite')
        predict_fn = load_crfsuite_tagger(model_path)
        prefix = 'crfsuite'
    else:
        model_path = os.path.join(root, 'models', 'crf_scratch_model.json')
        predict_fn = load_scratch_tagger(model_path)
        prefix = 'scratch'

    print('Model: %s' % model_path)
    print('Loading test data...')
    test_w, test_t = load_raw(test_words, test_tags)
    print('Loaded %d test sentences' % len(test_w))

    print('Running predictions...')
    correct = 0
    total = 0
    all_predictions = []

    for ws, gold in zip(test_w, test_t):
        pred = predict_fn(ws)
        all_predictions.append((ws, gold, pred))
        for p, g in zip(pred, gold):
            if p == g:
                correct += 1
            total += 1

    accuracy = correct / total * 100 if total > 0 else 0
    print('Accuracy: %d/%d = %.2f%%' % (correct, total, accuracy))

    tag_rows = compute_per_tag_metrics(all_predictions)
    conf_tags, conf_matrix = compute_confusion(all_predictions)

    conll_path = os.path.join(results_dir, '%s_predictions.conll' % prefix)
    write_conll(conll_path, all_predictions)
    print('Predictions: %s' % conll_path)

    compare_path = os.path.join(results_dir, '%s_comparison.txt' % prefix)
    write_comparison(compare_path, all_predictions)
    print('Comparison: %s' % compare_path)

    per_tag_path = os.path.join(results_dir, '%s_per_tag_f1.txt' % prefix)
    write_per_tag(per_tag_path, tag_rows)
    print('Per-tag F1: %s' % per_tag_path)

    confusion_path = os.path.join(results_dir, '%s_confusion.txt' % prefix)
    write_confusion(confusion_path, conf_tags, conf_matrix)
    print('Confusion: %s' % confusion_path)

    macro_f1 = sum(r[3] for r in tag_rows) / len(tag_rows) if tag_rows else 0
    total_support = sum(r[4] for r in tag_rows)
    weighted_f1 = sum(r[3] * r[4] for r in tag_rows) / total_support if total_support else 0

    metrics_path = os.path.join(results_dir, '%s_metrics.txt' % prefix)
    with open(metrics_path, 'w') as f:
        f.write('Model: %s\n' % args.model)
        f.write('Model path: %s\n' % model_path)
        f.write('Test sentences: %d\n' % len(test_w))
        f.write('Correct: %d\n' % correct)
        f.write('Total: %d\n' % total)
        f.write('Accuracy: %.2f%%\n' % accuracy)
        f.write('Macro F1: %.4f\n' % macro_f1)
        f.write('Weighted F1: %.4f\n' % weighted_f1)
    print('Metrics: %s' % metrics_path)


if __name__ == '__main__':
    main()
