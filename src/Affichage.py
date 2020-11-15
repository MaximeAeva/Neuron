import matplotlib.pyplot as plt
import networkx as nx

def loadMyNet(parms, layers_dims):
    G=nx.Graph()
    deep = 0
    for ld in layers_dims:
        for k in range(-ld//2, ld//2):
            G.add_nodes_from([k, deep])
        deep = deep + 1
            