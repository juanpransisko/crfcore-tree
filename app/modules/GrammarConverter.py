"""
Chomsky Normal Form (CNF) converter for context-free grammars.

Converts an arbitrary CFG into CNF so that it can be used with the
CYK parsing algorithm. The conversion handles:
    - Rules with more than two symbols on the right-hand side
    - Mixed terminal/non-terminal rules
    - Unit productions (A -> B)

The OHTree X-bar grammar for Filipino is converted through this
module before being fed to the CYK parser.
"""

RULE_DICT = {}


def read_grammar(grammar_file):
    """Read a grammar file and split each rule into a list of symbols."""
    with open(grammar_file) as cfg:
        lines = cfg.readlines()
    return [x.replace("->", "").split() for x in lines]


def add_rule(rule):
    """Register a rule in the global rule dictionary."""
    global RULE_DICT

    if rule[0] not in RULE_DICT:
        RULE_DICT[rule[0]] = []
    RULE_DICT[rule[0]].append(rule[1:])


def convert_grammar(grammar):
    """Convert a CFG to Chomsky Normal Form."""
    global RULE_DICT
    RULE_DICT = {}
    unit_productions, result = [], []
    res_append = result.append
    index = 0

    for rule in grammar:
        new_rules = []
        if len(rule) == 2 and rule[1][0] != "'":
            unit_productions.append(rule)
            add_rule(rule)
            continue
        elif len(rule) > 2:
            terminals = [(item, i) for i, item in enumerate(rule) if item[0] == "'"]
            if terminals:
                for item in terminals:
                    rule[item[1]] = f"{rule[0]}{str(index)}"
                    new_rules += [f"{rule[0]}{str(index)}", item[0]]
                index += 1
            while len(rule) > 3:
                new_rules += [f"{rule[0]}{str(index)}", rule[1], rule[2]]
                rule = [rule[0]] + [f"{rule[0]}{str(index)}"] + rule[3:]
                index += 1
        add_rule(rule)
        res_append(rule)
        if new_rules:
            res_append(new_rules)
    while unit_productions:
        rule = unit_productions.pop()
        if rule[1] in RULE_DICT:
            for item in RULE_DICT[rule[1]]:
                new_rule = [rule[0]] + item
                if len(new_rule) > 2 or new_rule[1][0] == "'":
                    res_append(new_rule)
                else:
                    unit_productions.append(new_rule)
                add_rule(new_rule)
    return result
