"""
TreeCo web application for Filipino treebank construction.

Provides a Flask-based interface for the two-stage NLP pipeline:
    1. POS tagging via the CRFPOST model (python-crfsuite)
    2. Constituency parsing via the CYK algorithm with the
       OHTree X-bar grammar in Chomsky Normal Form

The tagger uses a pre-trained CRF model to assign MGNN POS tags.
Tagged output is then fed to the CYK parser, which produces a
bracketed constituent tree following the OHTree grammar rules.
"""

import os
import sys

from flask import Flask, render_template, jsonify, abort
import json

from modules import treeco

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TREES_DIR = os.path.join(BASE_DIR, 'trees')
POS_MODELS_DIR = os.path.join(BASE_DIR, 'pos_models')
GRAMMARS_DIR = os.path.join(BASE_DIR, 'grammars')
MODULES_DIR = os.path.join(BASE_DIR, 'modules')

_tagger = None

def get_tagger():
	global _tagger
	if _tagger is None:
		try:
			import pycrfsuite
			_tagger = pycrfsuite.Tagger()
			model_path = os.path.join(POS_MODELS_DIR, 'crfpost_model.crfsuite')
			_tagger.open(model_path)
		except Exception as e:
			print("Failed to load CRF model: %s" % e, file=sys.stderr)
			return None
	return _tagger


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


def tag_sentence(sentence_text):
	"""POS-tag a sentence using the trained CRF model."""
	tagger = get_tagger()
	if tagger is None:
		return None

	tokens = sentence_text.strip().split()
	if not tokens:
		return []

	features = [word_features(tokens, i) for i in range(len(tokens))]
	tags = tagger.tag(features)
	return list(zip(tokens, tags))


def safe_filename(name):
	"""Sanitize a filename to prevent directory traversal."""
	basename = os.path.basename(name)
	if not basename or basename.startswith('.'):
		return None
	return basename


@app.route('/')
def index():
	return render_template('views/index.html')


@app.route('/simulation/')
def simulation():
	return render_template('views/simulation.html')


@app.route('/documentation/')
def documentation():
	return render_template('views/documentation.html')


@app.route('/save-tree/<string:tree>/<string:parsed_data>/')
def save_tree(tree, parsed_data):
	safe_tree = safe_filename(tree)
	if safe_tree is None:
		abort(400)

	tree_path = os.path.join(TREES_DIR, safe_tree)
	with open(tree_path, "a+", encoding="utf-8") as f:
		f.write(parsed_data + "\n")

	return "success"


@app.route('/inittreebanks/', methods=['GET'])
def init_trees():
	trees = []
	for f in os.listdir(TREES_DIR):
		name = os.path.splitext(f)[0]
		trees.append({'text': name, 'value': f})
	return jsonify(trees)


@app.route('/initmodels/', methods=['GET'])
def init_models():
	models = []
	for f in os.listdir(POS_MODELS_DIR):
		name = os.path.splitext(f)[0]
		models.append({'text': name, 'value': f})
	return jsonify(models)


@app.route('/initgrammars/', methods=['GET'])
def init_grammars():
	grammars = []
	for f in os.listdir(GRAMMARS_DIR):
		name = os.path.splitext(f)[0]
		grammars.append({'text': name, 'value': f})
	return jsonify(grammars)


@app.route('/treeco-manual-input/no-pos-end/with-file/<string:sentence>/<string:pos_model>/<string:pos_start_del>/<string:output_file>/', methods=['GET'])
def manual_input_no_pos_end_with_file(sentence, pos_model, pos_start_del, output_file):
	return _do_tagging(sentence, pos_start_del, "", output_file)


@app.route('/treeco-manual-input/no-pos-end/without-file/<string:sentence>/<string:pos_model>/<string:pos_start_del>/', methods=['GET'])
def manual_input_no_pos_end_without_file(sentence, pos_model, pos_start_del):
	return _do_tagging(sentence, pos_start_del, "", None)


@app.route('/treeco-manual-input/with-pos-end/with-file/<string:sentence>/<string:pos_model>/<string:pos_start_del>/<string:pos_end_del>/<string:output_file>/', methods=['GET'])
def manual_input_with_pos_end_with_file(sentence, pos_model, pos_start_del, pos_end_del, output_file):
	return _do_tagging(sentence, pos_start_del, pos_end_del, output_file)


@app.route('/treeco-manual-input/with-pos-end/without-file/<string:sentence>/<string:pos_model>/<string:pos_start_del>/<string:pos_end_del>/', methods=['GET'])
def manual_input_with_pos_end_without_file(sentence, pos_model, pos_start_del, pos_end_del):
	return _do_tagging(sentence, pos_start_del, pos_end_del, None)


def _do_tagging(sentence, pos_start_del, pos_end_del, output_file):
	"""Tag a sentence and write output files for the CYK parser."""
	tagged_pairs = tag_sentence(sentence)
	if tagged_pairs is None:
		abort(500)

	output_parts = []
	for word, tag in tagged_pairs:
		output_parts.append(word + pos_start_del + tag + pos_end_del + " ")
	output = "".join(output_parts)

	conll_path = os.path.join(MODULES_DIR, "tagged_output.txt")
	with open(conll_path, "w", encoding="utf-8") as f:
		for word, tag in tagged_pairs:
			f.write("%s\t%s\n" % (word, tag))

	if output_file:
		safe_out = safe_filename(output_file)
		if safe_out is None:
			abort(400)
		outputs_dir = os.path.join(BASE_DIR, 'outputs')
		os.makedirs(outputs_dir, exist_ok=True)
		with open(os.path.join(outputs_dir, safe_out), "w", encoding="utf-8") as f:
			f.write(output)

	crfpost_path = os.path.join(MODULES_DIR, "crfpost_output.txt")
	with open(crfpost_path, "w", encoding="utf-8") as f:
		f.write(output)

	return output


@app.route('/parse_data/', methods=['GET'])
def parse_data():
	tagged_path = os.path.join(MODULES_DIR, 'tagged_output.txt')
	treeco.normalize(tagged_path)
	tree = treeco.cyk_parsing()
	return tree


if __name__ == "__main__":
	app.run(debug=True)
