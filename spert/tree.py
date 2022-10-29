"""
Basic operations on trees.
"""

import numpy as np


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """

    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def head_to_tree(head, len_):
    """
    Convert a sequence of head indexes into a tree object.
    head: ["2", "6", "2", "3", "4", "0", "9", "9", "6", "9", "13", "13", "10", "13", "16", "14", "6"]
    len_: 控制句子的长度
    """
    head = head[:len_]
    root = None

    nodes = [Tree() for _ in head]

    for i in range(len(nodes)):
        h = head[i]  # 2
        nodes[i].idx = i  # 第i个node的idx为i
        nodes[i].dist = -1  # just a filler
        if h == 0:
            root = nodes[i]
        else:
            nodes[h - 1].add_child(nodes[i])

    assert root is not None
    return root


def tree_to_adj(sent_len, tree, directed=True):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    sent_len: 该batch中的句子的最大长度
    tree:所有句子head构建的tree
    directed:是否考虑方向
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        # 层次遍历
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    return ret


def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret
