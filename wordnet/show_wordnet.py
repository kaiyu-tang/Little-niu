#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
# @Time    : 2018/9/5 上午10:44  
# @Author  : Kaiyu  
# @Site    :   
# @File    : show_wordnet.py

from anytree import Node, RenderTree
from nltk.corpus import wordnet as wn

udo = Node("Udo")
marc = Node("Marc", parent=udo)
lian = Node("Lian", parent=marc)
dan = Node("Dan", parent=udo)
jet = Node("Jet", parent=dan)
jan = Node("Jan", parent=dan)
joe = Node("Joe", parent=dan)
dog = wn.synsets("dog")
print(dog)
c = dog[0]

print(dog[0].hypernyms())
print(dog[0].hyponyms())
for pre, fill, node in RenderTree(udo):
     print("%s%s" % (pre, node.name))

# import networkx as nx
# from graphviz import Digraph
#
# def closure_graph(synset, fn):
#     seen = set()
#     graph = Digraph()
#
#     def recurse(s):
#         if not s in seen:
#             seen.add(s)
#             graph.add_node(s.name)
#             for s1 in fn(s):
#                 graph.add_node(s1.name)
#                 graph.add_edge(s.name, s1.name)
#                 recurse(s1)
#
#     recurse(synset)
#     return graph
#
# dog = wn.synset('dog.n.01')
# graph = closure_graph(dog, lambda s: s.hypernyms())
# nx.draw_graphviz(graph)
