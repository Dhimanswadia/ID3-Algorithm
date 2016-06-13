from math import log2


class Dataset(object):
    """Represents a tabular, tuple-based dataset with ad-hoc functionality,
    supporting only ID3-related methods."""

    def __init__(self, data_list, is_formatted=True):
        """Initialises the data set, with is_formatted as an assumption
        that the .txt file with '\t'-separated values has headers in its
        first line with the target attribute as the last column of the
        headers."""
        self.data = []
        self.headers = None
        self.target = None
        self.tree = None
        self.ruleset = None
        self._is_formatted = is_formatted
        self.populate_data(data_list)
        print("Initiating dataset...\nHeaders: {}\nTarget: {}".format
              (", ".join(self.headers), self.target))

    def populate_data(self, data_list):
        """Populates the dataset with the given pre-processed list of
        lines read from the .txt file. Each tuple is stored as a
        dictionary."""
        headers = data_list[0].rstrip().split('\t')
        if self._is_formatted:
            self.headers = headers[:-1]
            self.target = headers[-1]
        for tup in data_list[1:]:
            tup = tup.rstrip().split('\t')
            tup = {k: v for k, v in zip(self.headers + [self.target], tup)}
            self.data.append(tup)

    def entropy(self, r, s=None, dataset=None):
        """Calculates the entropy of an attribute r, or attributes
        r and s, with r acting as the target attribute and s as the
        splitting attribute.
        """
        if not dataset:
            dataset = self.data
        if not s:
            values = [tup[r] for tup in dataset]
            count = {val: values.count(val) for val in values}
            entropy = 0
            for val in count.keys():
                p = count[val] / len(values)
                entropy -= p * log2(p)
            return entropy
        else:
            s_values = [tup[s] for tup in dataset]
            s_count = {val: s_values.count(val) for val in s_values}
            entropy = 0
            for val in s_count.keys():
                p = s_count[val] / len(s_values)
                subset = [tup for tup in dataset if tup[s] == val]
                entropy += p * self.entropy(r, dataset=subset)
            return entropy

    def information_gain(self, r, s):
        """Calculates the information gain (decrease in entropy) after
        splitting target attribute r with splitting attribute s."""
        return self.entropy(r) - self.entropy(r, s)

    def _id3(self, target, attrs, dataset):
        """Recursively constructs a decision tree given a target
        attribute, a list of predicting attributes, and an example
        dataset. Returns a tree of Node objects."""
        t_values = [tup[target] for tup in dataset]
        if len(set(t_values)) == 1:
            val = set(t_values).pop()
            return Node({'attr': target, 'val': val}, 'label')
        elif not attrs:
            common = max(set(t_values), key=t_values.count)
            return Node({'attr': target, 'val': common}, 'label')
        else:
            info_gain = {a: self.information_gain(target, a) for a in attrs}
            attr = max(attrs, key=lambda x: info_gain[x])
            root = Node(attr, 'attr')
            a_values = sorted(set([tup[attr] for tup in dataset]))
            for val in a_values:
                child = root.add(Node(val, 'val'))
                subset = [tup for tup in dataset if tup[attr] == val]
                if not subset:
                    subset_t_values = [tup[target] for tup in subset]
                    common = max(set(subset_t_values), key=subset_t_values.count)
                    child.add(Node({'attr': target, 'val': common}, 'label'))
                else:
                    if attrs and attr in attrs:
                        attrs.remove(attr)
                    child.add(self._id3(target, attrs, subset))

            return root

    def build_tree(self):
        self.tree = self._id3(self.target, self.headers, self.data)
        return self.tree

    def derive_ruleset(self):
        if self.tree is None:
            self.build_tree()
        self.ruleset = [rule for rule in self.paths(self.tree)]

    def print_ruleset(self):
        if self.ruleset is None:
            self.derive_ruleset()
        print("\nRuleset: ")
        for rule in self.ruleset:
            pred, result = rule[:-1], rule[-1]
            attr_pairs = tuple(zip(*(iter(pred),) * 2))
            rulestr = 'IF '
            for i in attr_pairs:
                rulestr += "({} == {}) AND ".format(i[0], i[1])
            rulestr = rulestr[:-4] + "THEN " + "{} = {}".format(
                result['attr'], result['val'])
            print(rulestr)

    def accuracy(self):
        if self.tree is None:
            raise ValueError("tree is empty")
        for rule in self.ruleset:
            pass

    def paths(self, node, path=None):
        if path is None:
            path = []
        if node.is_leaf:
            yield path + [node.value]
        for child in node.children:
            for leaf_path in self.paths(child, path + [node.value]):
                yield leaf_path


class Node(object):
    """Node of a decision tree, provides basic functionality
    of representing a decision tree with arbitrary labels
    and number of children."""

    def __init__(self, value, type, children=None):
        self.value = value
        self.type = type
        self.children = [] if children is None else children

    def __str__(self):
        if self.type == 'attr':
            return str(self.value)
        elif self.type == 'val':
            return str(self.value)
        elif self.type == 'label':
            attr = str(self.value['attr'])
            val = str(self.value['val'])
            return "{}: {}".format(attr, val)

    @property
    def is_leaf(self):
        return self.type == 'label'

    def add(self, child):
        self.children.append(child)
        return child

    def print(self, lvl=0, tab_lvl=0):
        print('\t'*lvl + '  \_'*tab_lvl + str(self))
        if tab_lvl:
            lvl += 1
        if not tab_lvl:
            tab_lvl += 1
        for child in self.children:
            child.print(lvl, tab_lvl)


if __name__ == '__main__':
    with open('data_titanic.txt') as fd:
        f = fd.readlines()
    d = Dataset(f)
    d.build_tree().print()
    d.print_ruleset()
