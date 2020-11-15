import matplotlib.pyplot as plt
import networkx as nx

def loadMyNet(parms, layers_dims):
    plt.figure(figsize=(20,10), dpi=200)
    G=nx.DiGraph()
    L = len(layers_dims)
    deep = 0
    node = 0
    pos = {}
    for i in range(L):
        ld = layers_dims[i][0]
        for k in range(1, ld+1):
            pos[node] = (deep, ((ld+1)/2)-k)
            node = node+1
        deep = deep + 1
        
    n = 0
    nn = 0
    cmap = plt.cm.jet 
    for l in range(1, L):
        nn = nn + layers_dims[l-1][0]
        for i in range(parms["W"+str(l)].shape[0]):
            for j in range(parms["W"+str(l)].shape[1]):
                w = abs(parms["W"+str(l)][i, j])
                G.add_edge(n+j, nn+i, color=cmap(w),weight=w)
        n = n + layers_dims[l-1][0]
        
    G.add_nodes_from(pos.keys())
    edges=G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    weights = [G[u][v]['weight'] for u,v in edges]
    
    nx.draw(G, pos, edges=edges, edge_color=colors, width=weights, node_size=10)
    
    plt.show()