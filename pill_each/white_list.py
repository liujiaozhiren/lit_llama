import torch

from tokenizer import Tokenizer


class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_point = 0
        self.word_num = 0


class White_list:
    def __init__(self, poi_list, tokenizer: Tokenizer, vocab_size=32000, suffix_constrain=False):
        # poi_list = [(poi_id, cat, token_len, lon, lat),...]
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.cat_tokens_list = []
        tmp_dict = {}
        for poi in poi_list:
            cat_name = poi[1]
            if suffix_constrain:
                cat_name = cat_name + "\"."
            poi_id = poi[0]
            self.cat_tokens_list.append((poi_id, tokenizer.encode(cat_name, bos=False)))
        self.tree_root = build_prefix_tree(self.cat_tokens_list)

        self.label_dict = self.get_label_dict(poi_list, suffix_constrain)
        # print(self.label_dict)

    def label(self, cat_name):
        assert cat_name in self.label_dict
        return self.label_dict[cat_name]
    def get_label_dict(self, poi_list, suffix_constrain=False):  # str->
        label_dict = {}
        for poi in poi_list:
            cat_name = poi[1]
            if cat_name in label_dict:
                continue
            if suffix_constrain:
                cat_name = cat_name + "\"."
            label = self.get_label(cat_name)
            label_dict[poi[1]] = label
        return label_dict

    def get_label(self, target):
        target_token = self.tokenizer.encode(target, bos=False)
        node = self.tree_root
        white_list_label = torch.zeros((len(target_token), self.vocab_size))
        for i, token_idx in enumerate(target_token):
            token_idx = int(token_idx)
            assert token_idx in node.children
            denominator = node.word_num
            for idx in node.children.keys():
                child_node = node.children[idx]
                numerator = child_node.word_num
                white_list_label[i, idx] = numerator / denominator
            node = node.children[token_idx]
        return white_list_label


def update_word_num(node):
    if node.end_point != 0:
        node.word_num = 1
    for child_node in node.children.values():
        update_word_num(child_node)
        node.word_num += child_node.word_num


def build_prefix_tree(word_list):
    root = TrieNode()
    for i, word in enumerate(word_list):
        poi_id, word = word
        node = root
        for bit in word:
            bit = int(bit)
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]
        node.end_point = node.end_point + 1
    update_word_num(root)
    return root
