import numpy as np
import pickle
import os
import re
import networkx as nx
import pandas as pd
from networkx.algorithms.shortest_paths.generic import shortest_path_length
from nltk.corpus import stopwords
from helper_files.py import *



def gen_graph(data):
    G = nx.Graph()
    stopWords = set(stopwords.words('english'))
    for d in data:
        reviews = d[0]
        for s in reviews:
            prev_word = None
            for word in s:
                if word not in stopWords:
                    if not word.isnumeric():
                        if len(word) >= 3:
                            if not word in list(G.nodes):
                                G.add_node(word)
                            if prev_word:
                                if not (prev_word,word) in list(G.edges):
                                    G.add_edge(prev_word,word)
                            prev_word = word
                    else:
                        if not word in list(G.nodes):
                                G.add_node(word)
                        if prev_word:
                            if not (prev_word,word) in list(G.edges):
                                G.add_edge(prev_word,word)
                        prev_word = word
    largest_cc = max(nx.connected_components(G), key = len)
    G = G.subgraph(largest_cc)
    return(G)

def findResSet(T):
    degs = T.degree()
    root = 0
    path = -1
    for k in degs:
        if k[1] > 1: root = k[0]
        if k[1] > 2:
            path = -1
            break
        if k[1] <= 1: path = k[0]
    if path != -1: return ({}, [path])

    (L, um, _) = partitionLeaves(T, node=root)
    R = []
    for h in L:
        if len(L[h]) > 0:
            i = np.random.choice(len(L[h]))
            R.extend(L[h][:i] + L[h][i+1:])
    return (L, R)

def partitionLeaves(T, node=0,  parent=-1):
    (marked, unmarked, last) = ({}, [], -1)
    N = T.degree(node)
    if N == 1: return (marked, [node], last)
    if N >= 3:
        marked[node] = []
        last = node
    children = [n for n in T.neighbors(node) if n!=parent]
    for child in children:
        (m, um, l) = partitionLeaves(T, node=child, parent=node)
        marked.update(m)
        if len(um) > 0 and N >= 3: marked[node].extend(um)
        else:
            unmarked.extend(um)
            last = l 
    if len(unmarked) > 0 and last != -1: marked[last].extend(unmarked)
    return (marked, unmarked, last)

#Intersect multiple spanning tree approach
def find_trees(G,root):
    btree = nx.bfs_tree(G,source = root)
    dtree = nx.dfs_tree(G,source = root)
    btree = btree.to_undirected()
    dtree = dtree.to_undirected()
    return(btree,dtree)

def gen_trees_sets(G, num_trees = 1, union = False):
    roots = np.random.choice(list(G.nodes), num_trees, replace = False)
    res_set = None
    for root in roots:
        btree = nx.bfs_tree(G,source = root)
        btree = btree.to_undirected()
        L,R = findResSet(btree)
        R = set(R)
        if res_set:
            if union:
                res_set = res_set | R
            else:
                res_set = res_set & R
        else:
            res_set = R
    return(list(res_set))

def distanceMatrix(G, R = None):
    if(R):
        #set up so that resolving set R will be column labels and rest of graph will be rows
        dist = {}
        for r in R:
            dist.update({r : dict(shortest_path_length(G,source = r))})
    else:
        dist = dict(shortest_path_length(G))
    dist_mat = pd.DataFrame.from_dict(dist)
    return(dist_mat)

def checkResolving(M):
    n = M.shape[0]
    for i in range(0,n-1):
        for j in range(i+1,n):
            if all(M.iloc[i,:] == M.iloc[j,:]):
                print("Row {} is equal to Row {}".format(i,j))
                return(False)
    return(True)
    
# def prune_ResSet(G,R,i = 0,removed = []):
#     M = distanceMatrix(G,R = R)
#     new_R = R
#     M.drop(index = R) #Remove resolving set from check since we know those will be resolved
#     np.random.shuffle(new_R) #Shuffle R
#     is_resolving = True
#     print("iteration {}. Size of Resolving set {}".format(i,len(R)))
#     t = np.randint(0,len(new_R)-1)
#     r = new_R[t]
#     R.remove(r)
#     i = i+1
#     M.drop(columns = r)#Remove node from resolving set
#     M.drop(index = R)
#     is_resolving = checkResolving(M)
#     if is_resolving:
#         (result,new_R) = prune_ResSet(G,R,i=i)
#         if result:
#             R = result
#     return(is_resolving,R)

directory = "reviews_3mo"
result = read_dir(directory)
G = gen_graph(result)
graph_data = (list(G.nodes),list(G.edges))
with open("graph_data.graph","wb") as f:
    pickle.dump(graph_data,f)

print("number of unique words across all docs: {}".format(len(G.nodes)))
R = gen_trees_sets(G,num_trees = 3, union = False)
print("Size of resolving set R: {}".format(len(R)))
M = distanceMatrix(G,R = R)
is_resolving = checkResolving(M)
if (is_resolving):
    print("Set R is resolving")
    with open("working_R.set","wb") as f:
        pickle.dump(R,f)