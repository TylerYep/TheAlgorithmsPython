"""
    disjoint set
    Reference: https://en.wikipedia.org/wiki/Disjoint-set_data_structure
"""


class Node:
    def __init__(self, data):
        self.data = data


def make_set(x):
    """
    make x as a set.
    rank is the distance from x to its' parent
    root's rank is 0
    """
    x.rank = 0
    x.parent = x


def union_set(x, y):
    """
    union two sets.
    set with bigger rank should be parent, so that the
    disjoint set tree will be more flat.
    """
    x, y = find_set(x), find_set(y)
    if x.rank > y.rank:
        y.parent = x
    else:
        x.parent = y
        if x.rank == y.rank:
            y.rank += 1


def find_set(x):
    """
    return the parent of x
    """
    if x != x.parent:
        x.parent = find_set(x.parent)
    return x.parent
