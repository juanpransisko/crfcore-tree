"""
TreeCo core module: CYK parsing and Filipino morphological processing.

Implements the Cocke-Younger-Kasami (CYK) constituency parsing
algorithm over grammars in Chomsky Normal Form (CNF). The parser
operates on POS-tagged input produced by the CRFPOST tagger and
uses the OHTree X-bar grammar for Filipino.

The normalization step prepares tagged output for parsing by:
    - Appending lexical rules (TAG -> 'word') to the grammar
    - Tracing Filipino verb and adjective morphological affixes
    - Removing punctuation and function words not in the grammar

References:
    Kasami, T. (1966). An Efficient Recognition and Syntax-Analysis
        Algorithm for Context-Free Languages. AFCRL-65-758.
    Younger, D. H. (1967). Recognition and Parsing of Context-Free
        Languages in Time n^3. Information and Control, 10(2).
"""

import sys, os, os.path, subprocess
from random import random
from math import exp
import shutil

try:
	from .feature_function import FeatureFunction
except (SystemError, ImportError):
	from feature_function import FeatureFunction

try:
	from . import GrammarConverter
except (SystemError, ImportError):
	import GrammarConverter

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_MODULE_DIR)


class Node:
	def __init__(self, symbol, y, x, i, terminal_index = None, y2 = None, x2 = None, i2 = None):
		if terminal_index is not None:
			self.terminal_index = terminal_index
		self.child1 = [(y, x, i)]
		self.child2 = [(y2, x2, i2)]
		self.symbol = symbol

	def __repr__(self):
		return self.symbol


class Parser:
	def __init__(self, grammar, sentence):
		self.parse_table = None
		self.prods = {}
		self.grammar = None
		if os.path.isfile(grammar):
			self.grammar_from_file(grammar)
		else:
			self.grammar_from_string(grammar)

		self.__call__(sentence)

	def __call__(self, sentence, parse = False):
		if os.path.isfile(sentence):
			with open(sentence) as inp:
				self.input = inp.readline().split()
				if parse:
					self.parse()
		else:
			self.input = sentence.split()


	def grammar_from_file(self, grammar):
		self.grammar = GrammarConverter.convert_grammar(GrammarConverter.read_grammar(grammar))

	def grammar_from_string(self, grammar):
		self.grammar = GrammarConverter.convert_grammar(
            [x.replace("->", "").split() for x in grammar.split("\n")])

	def parse(self):
		"""Run the CYK algorithm to fill the parse table."""
		length = len(self.input) + 1
		self.parse_table = [[[] for x in range(length)] for y in range(length - 1)]

		for i in range(1, length):
			for rule in self.grammar:
				if f"'{self.input[i - 1]}'" == rule[1]:
					self.parse_table[i - 1][i].append(Node(rule[0], None, None, None, i - 1))
		for i in range(2, length):
			for j in range(i - 2, -1, -1):
				for k in range(j + 1, i):
					for rule in self.grammar:
						if (j, k) not in self.prods:
							self.prods[j, k] = [node.symbol for node in self.parse_table[j][k]]
						prod1 = self.prods[j, k]
						if (k, i) not in self.prods:
							self.prods[k, i] = [node.symbol for node in self.parse_table[k][i]]
						prod2 = self.prods[k, i]

						if rule[1] in prod1 and rule[2] in prod2:
							node = Node(rule[0], j, k, prod1.index(rule[1]), None, k, i,
										prod2.index(rule[2]))
							self.parse_table[j][i].append(node)
							if (j, i) not in self.prods:
								self.prods[j, i] = []
							self.prods[j, i].append(node.symbol)
							if len(prod1) > 1:
								for ind, symbol in enumerate(prod1):
									if symbol == rule[1] and ind != prod1.index(rule[1]):
										node.child1.append((j, k, ind))
							if len(prod2) > 1:
								for ind, symbol in enumerate(prod2):
									if symbol == rule[2] and ind != prod2.index(rule[2]):
										node.child2.append((k, i, ind))

	def print_tree(self, output = True):
		"""Extract the parse tree from the completed parse table."""
		cell = self.parse_table[0][len(self.input)]
		if not cell:
			return "Input cannot be parsed."
		start_symbol = cell[0]
		if start_symbol.symbol == self.grammar[0][0]:
			trees = self.generate_trees(start_symbol)
			if trees.find(f"){start_symbol.symbol}") > -1:
				# There are multiple parses. This formats the string properly.
				trees = trees.replace(f"){start_symbol.symbol}", f")\n({start_symbol.symbol}")
			parse_output = os.path.join(_MODULE_DIR, "parse_tree_output.txt")
			with open(parse_output, "w+", encoding="utf-8") as f:
				f.write(trees)
				return trees if output else trees
		else:
			return "Input cannot be parsed."

	def generate_trees(self, node):
		"""Recursively build bracketed tree string from parse table nodes."""
		tree, table = "( ", self.parse_table
		for child1 in node.child1:
			child1_y, child1_x, child1_i = child1
			for child2 in node.child2:
				child2_y, child2_x, child2_i = child2
				if child1_x is None and child1_y is None and child2_x is None and child2_y is None:
					tree = f"{tree}{node.symbol} '{self.input[node.terminal_index]}'"
				else:
					tree = f"{tree}{node.symbol} "
				if child1_y is not None and child1_x is not None:
					tree = f"{tree}{self.generate_trees(table[child1_y][child1_x][child1_i])}"
				if child2_y is not None and child2_x is not None:
					tree = f"{tree}{self.generate_trees(table[child2_y][child2_x][child2_i])}"
				tree += " ) "
				break # To parse just one tree
		return tree


def cyk_parsing():
	grammar = os.path.join(_MODULE_DIR, "grammar.txt")
	sentence = os.path.join(_MODULE_DIR, "normalized_input.txt")
	cyk = Parser(grammar, sentence)
	cyk.parse()
	tree = cyk.print_tree()
	tree_output = os.path.join(_MODULE_DIR, "tree_output.txt")
	with open(tree_output, "w", encoding="utf-8") as f:
		f.write(tree)
	return tree


def load_data(file_path):
	"""Read the full content of a corpus file."""
	file = open(file_path, encoding = "cp850")
	data = file.read()
	file.close()
	return data


def segment(data):
	"""Split corpus data into non-empty lines."""
	sentences = []
	sentences = data.split("\n")

	while '' in sentences:
		sentences.remove('')

	return sentences


def tokenize(data):
	"""Split a line into whitespace-delimited tokens."""
	tokens = data.split(" ")
	tokens = [t for t in tokens if t.strip()]
	return tokens


def get_tags(data):
	"""Collect the set of all POS tags occurring in the data."""
	available_tags = set()

	for words, tags in data:
		available_tags.update(tags)

	return list(available_tags)


def generate_initial_weights(count):
	"""Initialize random weights for feature functions."""
	return [random() for _ in range(count)]


def create_feature_functions(data):
	"""Build transition feature functions from tagged data."""

	feature_functions = set()

	for j, (words, tags) in enumerate(data):
		for i in range(1, len(tags)):
			feature_functions.add(FeatureFunction(tags[i - 1], tags[i]))

	return list(feature_functions)


def calculate_score(feature_functions, weights, tags):

	score = 0

	for i, (feature_function, weight) in enumerate(zip(feature_functions, weights)):

		for i in range(1, len(tags)):
			score += weight * feature_function.apply(tags[i - 1], tags[i])

	return score


def calculate_empirical_expectation(feature_function, tags):

	empirical_expectation = 0

	for i in range(1, len(tags)):
		empirical_expectation += feature_function.apply(tags[i - 1], tags[i])

	return empirical_expectation


def calculate_predicted_expectation(feature_function, data, pos_tags, weights, feature_functions, curr_tags):

	probability_given_words = 0
	feature_function_sum = 0
	predicted_expectation = 0

	for words, tags in data:
		probability_given_words += calculate_prob_tags_given_words(tags, feature_functions, weights, data)

	for feature_function in feature_functions:
		for i in range(len(curr_tags)):
			feature_function_sum += feature_function.apply(curr_tags[i-1], curr_tags[i])

	predicted_expectation = probability_given_words * feature_function_sum

	return predicted_expectation


def calculate_prob_tags_given_words(curr_tags, feature_functions, weights, data):

	numerator = 0
	denominator = 0

	numerator = exp(calculate_score(feature_functions, weights, curr_tags))

	for words, tags in data:
		denominator += exp(calculate_score(feature_functions, weights, tags))

	probability = numerator / denominator

	return probability


def load_model(model_path):

	content = load_data(model_path)
	sentences = segment(content)

	return sentences


def crf_training(pos_tags, feature_functions, data):
	"""Train CRF weights using gradient-based parameter estimation."""
	learning_rate = 0.5
	iterations = 10
	probability = 0
	score = 0

	weights = generate_initial_weights(len(feature_functions))

	for _ in range(iterations):
		for i, (feature_function, weight) in enumerate(zip(feature_functions, weights)):
			for j, (words, tags) in enumerate(data):

				empirical_expectation_value = calculate_empirical_expectation(feature_function, tags)

				predicted_expectation_value = calculate_predicted_expectation(feature_function, data, \
					pos_tags, weights, feature_functions, tags)

				weights[i] = weight + (learning_rate * (empirical_expectation_value - predicted_expectation_value))


def normalize(file_to_norm):
	"""Prepare tagged output for CYK parsing with the OHTree grammar."""
	conll = open(file_to_norm,"r", encoding="utf-8")
	arr_input = conll.readlines()
	arr_word = []
	arr_tag = []
	arr_temp = []

	while '\n' in arr_input:
		arr_input.remove('\n')

	i = 0
	for i in range(len(arr_input)):
		arr_temp = arr_input[i].split('\t')
		arr_word.append(arr_temp[0])
		arr_tag.append(arr_temp[1].rstrip())

	additional_grammar = ""
	i = 0
	for i in range(len(arr_input)):
		additional_grammar += arr_tag[i] + " -> \'" + arr_word[i] + "\'\n"

	grammar_src = os.path.join(_APP_DIR, 'grammars', 'OHTree_Grammar_beta_AA.txt')
	grammar_dst = os.path.join(_MODULE_DIR, 'grammar.txt')
	shutil.copyfile(grammar_src, grammar_dst)

	with open(grammar_dst, "a+", encoding="utf-8") as grammar_file:
		grammar_file.write("\n" + additional_grammar)

	new_words = []
	for i in range(len(arr_tag)):
		added_token = ""
		if arr_tag[i].startswith('V'):
			added_token = trace_vrb_aspect(arr_word[i])
		elif arr_tag[i].startswith('J'):
			added_token = trace_adj_aspect(arr_word[i])

		if added_token:
			new_words.append(added_token)
		new_words.append(arr_word[i])
	arr_word = new_words

	# Filter tokens not handled by the OHTree grammar
	norm_input = []
	arr_word2 = []
	arr_word3 = []
	arr_word4 = []
	arr_word5 = []

	i = 0
	for i in range(len(arr_word)):
		if(arr_word[i] != '.'):
			arr_word2.append(arr_word[i])

	i = 0
	for i in range(len(arr_word2)):
		if(arr_word2[i] != ','):
			arr_word3.append(arr_word2[i])

	i = 0
	for i in range(len(arr_word3)):
		if(arr_word3[i] != 'ng'):
			arr_word4.append(arr_word3[i])

	i = 0
	for i in range(len(arr_word4)):
		if(arr_word4[i] != 'mga'):
			arr_word5.append(arr_word4[i])

	i = 0
	for i in range(len(arr_word5)):
		if(arr_word5[i] != 'na'):
			norm_input.append(arr_word5[i])

	normalized_input = ""
	for n in norm_input:
		normalized_input += n + " "

	norm_path = os.path.join(_MODULE_DIR, "normalized_input.txt")
	with open(norm_path, "w", encoding="utf-8") as f:
		f.write(normalized_input)


def trace_adj_aspect(adj_word):
	"""Trace Filipino adjective morphological affixes."""
	added_token = ""

	if(adj_word.startswith('ma') or adj_word.startswith('Ma')):
		added_token = "[ma-|basic]"
	
	if(adj_word.endswith('an')):
		added_token = "[-an|adj_verb]"

	if(adj_word.startswith('napaka') or adj_word.startswith('Napaka')):
		added_token = "[napaka-|intensive]"

	if(adj_word.startswith('pinaka') or adj_word.startswith('Pinaka')):
		added_token = "[pinaka-|superlative]"

	if(adj_word.startswith('pagka')  or adj_word.startswith('Pagka')):
		added_token = "[pagka-|superlative]"

	return added_token


def trace_vrb_aspect(vrb_word):
	"""Trace Filipino verb morphological affixes for aspect marking."""
	added_token = ""

	if 'um' in vrb_word:
		added_token = "[-um-/m-|actor] [-um-|Infinitive]"
	
	if(vrb_word.endswith('ag')):
		added_token = "[-ag-|actor]"

	if(vrb_word.startswith('i')):
		added_token = "[i-|patient]"

	if(vrb_word.endswith('an')):
		added_token = "[-an|direction]"

	return added_token


def convert_to_semi_conll(file_words, file_output_name):
	"""Convert space-separated word sequences to one-word-per-line format."""
	content_words = load_data(file_words)
	sentences = segment(content_words)

	with open(file_output_name, "w+", encoding="utf-8") as file_output:
		for words_seq in sentences:
			token_words = tokenize(words_seq)
			output = ""
			for token in token_words:
				if token != " ":
					output += token + "\n"
			file_output.write(output + "\n")