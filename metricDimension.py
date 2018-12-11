#Python code implementing the ICH (Information Content Heuristic) approximation algorithm for metric dimension
#Note that this code also applies to multilateration problems, the algorithm does not use the fact that distance matrices for simple graphs are symmetric
#this file also includes code implementing the constructive hamming graph bound for metric dimension
#in addition, functions to generate and pick 3mer and 12mer resolving sets are included here

import numpy as np
import networkx as nx
from scipy.stats import entropy
from scipy.misc import comb
from itertools import product, combinations
import multiprocessing as mp

#######################
### SAVE/READ STATE ###
#######################
#save a snapshot of the algorithms progress (current tags and chosen columns)
#input: tags - a list of tags for each row based on the chosen columns
#       chosen - a list of the columns chosen so far
#       saveFile - name of the file to write to
#       overwrite - if true overwrite the contents of the file, otherwise append. default is False
def saveState(tags, chosen, saveFile, overwrite=False):
  with open(saveFile, 'w' if overwrite else 'a') as f:
    f.write('tags: '+','.join(tags)+'\n')
    f.write('chosen: '+','.join(map(str, chosen))+'\n')

#read a saved state of the algorithm 
#input: inFile - the file to read
def readState(inFile):
  tags = []
  chosen = []
  (c, t) = ('', '')
  with open(inFile, 'r') as f:
    i = 0
    for line in f:
      if line!='\n' and i % 2 == 1: c = line.rstrip()
      elif line!='\n': t = line.rstrip()
      i += 1
  return (t[6:].split(','), map(int, c[8:].split(',')))

#####################
### ICH ALGORITHM ###
#####################
#determine the joint entropy that would result from adding a given column
#input: col - the column to check
#       M - the distance matrix
#       tags - tags given the columns already chosen
#return: the joint entropy, the column, and the distribution of tags
def colEntropy(col, M, tags):
  jointDistr = {}
  for i in xrange(len(tags)):
    t = tags[i]+';'+str(M[i][col])
    if t not in jointDistr: jointDistr[t] = 0
    jointDistr[t] += 1
  e = entropy(jointDistr.values(), base=2)
  return (e, col, jointDistr)

#iterate over unchosen columns to determine the one which maximizes the change in entropy
#input: M - the distance matrix
#       tags - tages given the columns already chosen
#       check - a list of columns left to check
#       chosen - optional list of already chosen columns. defaults to empty
#       procs - optional argument specifying the number of processes to use. defaults to 1
#return: the tag distribution with the new column, the new tags, the entropy with the new column, an updated list of chosen columns
def pickColumn(M, tags, check, chosen=[], procs=1):
  (eMax, cMax, distr) = (-1, -1, {})
  if procs > 1:
    pool = mp.Pool(processes=procs)
    results = [pool.apply_async(colEntropy, args=(c, M, tags)) for c in check]
    results = [r.get() for r in results]
    (eMax, cMax, distr) = max(results)
  else:
    for col in check:
      (e, col, condDistr) = colEntropy(col, M, tags)
      if e > eMax: (eMax, cMax, distr) = (e, col, condDistr)
  chosen.append(cMax)
  for i in xrange(len(tags)):
    tags[i] += ';'+str(M[i][cMax])
  return (distr, tags, eMax, chosen)

#approximate the metric dimension given a distance matrix 
#follows the ICH algorithm
#input: M - the distance matrix
#       name - optional prefix to give to a file in which to save current state of the program. defaults to empty
#       read_state - optional name of a file to read current state from. default is empty
#       randOrder - optional boolean value. if true randomize the order in which columns are checked, defaults to true
#       procs - optional number of processes to run, defaults to 1
#return: a list of chosen columns representing a resolving set
def approxMetricDim(M, name='', read_state='', randOrder=True, procs=1):
  progress_file = name+'_progress.txt'
  distr = {}
  tags = ['' for _ in xrange(len(M))]
  chosen = []
  if read_state:
    (tags, chosen) = readState(read_state)
    for i in xrange(len(tags)):
      if tags[i] not in distr: distr[tags[i]] = []
      distr[tags[i]].append(i)
  elif name: saveState([], [], progress_file, overwrite=True)
  n = len(M[0])
  check = list(np.random.permutation(n)) if randOrder else range(n)
  H = 0
  while len(distr) < n and len(chosen) < n:#
    (distr, tags, H, chosen) = pickColumn(M, tags, check, chosen=chosen, procs=procs)
    check.remove(chosen[-1])
    if name: saveState(tags, chosen, progress_file, overwrite=True)
  if len(distr) < n and len(chosen) == n: print('No solution exists')#, distr)
  return chosen

##############################
### ICH ALGORITHM NETWORKX ###
##############################
#compute the metric dimension of a networkx graph using the ich algorithm
#includes option to save memory by computing distances to a single node when needed
#notice that functions in the previous section may be changed slightly to better accomodate the reqirements in this section
#we write totally separate functions just to make sure nothing breaks
#UNTESTED

#determine the joint entropy that would result from adding a given column
#input: col - the column to check
#       G - the graph
#       tags - tags given the columns already chosen
#       nodes - sorted list of nodes
#       distance - if -1, use the full distance matrix. if 1, use the adjacency matrix. if >1, use the minimum of the distance matrix and the given distance for each entry
#return: the joint entropy, the column, and the distribution of tags
def colGraphEntropy(col, G, tags, nodes, distance):
  jointDistr = {}
  D = []
  if distance==-1 or distance>1:
    D = nx.single_source_shortest_path_length(G, col, cutoff=(None if distance==-1 else distance))
    if distance==-1: D = [D.get(j, -1) for j in nodes]
    if distance>1: D = [min(D.get(j, distance), distance) for j in nodes]
  else: #adjacency matrix
    neighbors = G.neighbors(col)
    D = [1 if j in neighbors else 0 for j in nodes]
  for i in xrange(len(tags)):
    t = tags[i]+';'+str(D[i])
    if t not in jointDistr: jointDistr[t] = 0
    jointDistr[t] += 1
  e = entropy(jointDistr.values(), base=2)
  return (e, col, D, jointDistr)

#iterate over unchosen columns to determine the one which maximizes the change in entropy
#input: G - the graph
#       tags - tages given the columns already chosen
#       check - a list of columns left to check
#       chosen - optional list of already chosen columns. defaults to empty
#       distance - if -1, use the full distance matrix. if 1, use the adjacency matrix. if >1, use the minimum of the distance matrix and the given distance for each entry. default is -1
#       procs - optional argument specifying the number of processes to use. defaults to 1
#return: the tag distribution with the new column, the new tags, the entropy with the new column, an updated list of chosen columns
def pickGraphColumn(G, tags, check, chosen=[], distance=-1, procs=1):
  (eMax, cMax, dists, distr) = (-1, -1, [], {})
  nodes = sorted(G.nodes())
  if procs > 1:
    pool = mp.Pool(processes=procs)
    results = [pool.apply_async(colGraphEntropy, args=(c, G, tags, nodes, distance)) for c in check]
    results = [r.get() for r in results]
    (eMax, cMax, dists, distr) = max(results)
  else:
    for col in check:
      (e, col, D, condDistr) = colGraphEntropy(col, G, tags, nodes, distance)
      if e > eMax: (eMax, cMax, dists, distr) = (e, col, D, condDistr)
  chosen.append(cMax)
  for i in xrange(len(tags)):
    tags[i] += ';'+str(dists[i])
  return (distr, tags, eMax, chosen)

#approximate the metric dimension given a networkx graph
#follows the ICH algorithm
#input: G - the graph
#       fullMatrix - optional boolean indicating whether or not the compute the full matrix. default is true
#       distance - if -1, use the full distance matrix. if 1, use the adjacency matrix. if >1, use the minimum of the distance matrix and the given distance for each entry. default is -1
#       name - optional prefix to give to a file in which to save current state of the program. defaults to empty
#       read_state - optional name of a file to read current state from. default is empty
#       randOrder - optional boolean value. if true randomize the order in which columns are checked, defaults to true ####!!!!!! only works if fullMatrix=True at the moment
#       procs - optional number of processes to run, defaults to 1
#return: a list of chosen columns representing a resolving set
def approxGraphMetricDim(G, fullMatrix=True, distance=-1, name='', read_state='', randOrder=True, procs=1):
  check = sorted(G.nodes())
  if fullMatrix:
    D = []
    if distance==-1 or distance>1:
      D = nx.all_pairs_shortest_path_length(G, cutoff=(None if distance==-1 else distance))
      if distance==-1: D = [[D[i].get(j, -1) if i in D else -1 for j in check] for i in check]
      if distance>1: D = [[min(D[i].get(j, distance) if i in D else distance, distance) for j in check] for i in check]
    elif distance==1:
      D = [[1 if j in G.neighbors(i) else 0 for j in check] for i in check]
    return approxMetricDim(D, name=name, read_state=read_state, randOrder=randOrder, procs=procs)
  progress_file = name+'_progress.txt'
  distr = {}
  tags = ['' for _ in xrange(len(check))]
  chosen = []
  if read_state:
    (tags, chosen) = readState(read_state)
    for i in xrange(len(tags)):
      if tags[i] not in distr: distr[tags[i]] = []
      distr[tags[i]].append(i)
  elif name: saveState([], [], progress_file, overwrite=True)
  n = len(G)
  H = 0
  while len(distr) < n and len(chosen) < n:#
    (distr, tags, H, chosen) = pickGraphColumn(G, tags, check, chosen=chosen, distance=distance, procs=procs)
    check.remove(chosen[-1])
    if name: saveState(tags, chosen, progress_file, overwrite=True)
  if len(distr) < n and len(chosen) == n: print('No solution exists', distr)
  return chosen

#############################
### ICH ALGORITHM CLASSES ###
#############################
#approximate the metric dimension given a distance matrix considering collisions only between different groups or communities
#follows a modified version of the ICH algorithm
#NOTE: what's the right way to do this?

#######################
### BOUND ALGORITHM ###
#######################
#given a resolving set of a Hamming graph H(k, a), determine a resolving set for H(k+1, a)
#input: resSet - a resolving set for H(k, a)
#       alphabet - the alphabet from which to draw characters for the new resolving set
#       rand - optional boolean, if true randomize resSet and alphabet order. default is false
#return: a resolving set for H(k+1, a)
def hammingConstruction(resSet, alphabet, rand=False):
  if rand:
    resSet = list(np.random.permutation(resSet))
    alphabet = ''.join(list(np.random.permutation([a for a in alphabet])))
  newResSet = [r+alphabet[2*i] if 2*i<len(alphabet) else r+alphabet[0] for (r,i) in zip(resSet, xrange(len(resSet)))]#[r+alphabet[0] for r in resSet]
  num = len(alphabet) / 2
  for i in xrange(num):
    v = resSet[i]+alphabet[2*i+1]
    newResSet.append(v)
  return newResSet

###########################
### HAMMING BRUTE FORCE ###
###########################
#find all resovling sets of a Hamming graph via a brute force search for a particular size
#WARNING: this may be extremely slow even for small values of k and alphabet
#input: k - the length of strings in the hamming graph
#       alphabet - the alphabet to use in the hamming graph
#       size - the size of sets to check
#return: all resolving sets of the given size for the specified hamming graph
def hammingBruteForce(k, alphabet, size):
  resSets = []
  kmers = sorted([''.join(x) for x in product(alphabet, repeat=k)])
  D = [[hammingDist(a, b) for a in kmers] for b in kmers]
  numCombos = comb(len(D[0]), size)
  i = 0
  for cols in combinations(range(len(D[0])), size):
    if i%10000==0: print('Brute force progress: ', i / numCombos)
    i += 1
    if checkUnique(cols, D): resSets.append([kmers[c] for c in cols])
  return resSets

######################
### MISC FUNCTIONS ###
######################
#computes the hamming distance or number of mismatches between two strings
#if one string is longer the other, only its prefix is used
#input: a, b - two sequences to compare
#return: the hamming distance between a and b
def hammingDist(a, b):
  return sum([1 if x!=y else 0 for (x,y) in zip(a,b)])

#given a set of columns and a distance matrix, check that the set is resolving
#input: R - a set of columns
#       D - a distance matrix
#return: true if R is resolving and false otherwise
def checkUnique(R, D):
  tags = {}
  for r in xrange(len(D)):
    tag = ','.join([str(D[r][x]) for x in R])
    if tag in tags: return False
    tags[tag] = 1
  return True

#check that a given set of strings is resolving for a specified hamming graph
#input: R - a set of strings to check as resolving
#       k - length of strings
#       alphabet - characters that strings are composed of
#return: true if R is resolving on H(k, alphabet) and false otherwise
def checkUniqueHamming(R, k, alphabet):
  tags = {}
  i = 0
  tot = float(np.power(len(alphabet), k))
  for seq in product(alphabet, repeat=k):
    if i%10000==0: print('check unique progress: ', i/tot)
    i += 1
    seq = ''.join(seq)
    tag = ','.join(map(str, [hammingDist(seq, r) for r in R]))
    if tag not in tags: tags[tag] = 1
    else: return False
  return True

############
### MAIN ###
############
#determine resolving sets for H(3, 4) and H(12, 4), DNA 3mers and 12mers, using the ICH algorithm for 3mers and the constructive algorithm for 12mers
def findResSets():
  kmers = sorted([''.join(x) for x in product('ACGT', repeat=3)])
  D = [[hammingDist(a, b) for a in kmers] for b in kmers]
  R = approxMetricDim(D, name='', read_state='', procs=1)
  R = [kmers[r] for r in R]
  print('3mer resolving set via ICH', len(R), R)
  #['AAA', 'ACC', 'CAG', 'GGC', 'CGT', 'AAC']

  T = [r for r in R]
  for i in xrange(20):
    R = [t for t in T]
    while len(R[0]) < 7:
      R = hammingConstruction(R, 'ACGT', rand=True)
      if not checkUniqueHamming(R, len(R[0]), 'ACGT'):
        print('Construction failed', len(R[0]), R)
        break
      else: print('Success', i, len(R), len(R[0]), R)

  while len(R[0]) < 12:
    R = hammingConstruction(R, 'ACGT', rand=True)
    print(R)
    if not checkUniqueHamming(R, len(R[0]), 'ACGT'):
      print('Construction failed building 12-mer', len(R[0]), R)
      break
    else: print('Success', len(R), len(R[0]), R)
  print('12mer resolving set via construction', len(R), R)
  #['AAAAAAAAAAAA', 'ACCGGGGGGGGG', 'CAGAAAAAAAAA', 'GGCAAAAAAAAA', 'CGTAAAAAAAAA', 'AACAAAAAAAAA', 'AAACAAAAAAAA', 'ACCTAAAAAAAA', 'AAAACAAAAAAA', 'ACCGTAAAAAAA', 'AAAAACAAAAAA', 'ACCGGTAAAAAA', 'AAAAAACAAAAA', 'ACCGGGTAAAAA', 'AAAAAAACAAAA', 'ACCGGGGTAAAA', 'AAAAAAAACAAA', 'ACCGGGGGTAAA', 'AAAAAAAAACAA', 'ACCGGGGGGTAA', 'AAAAAAAAAACA', 'ACCGGGGGGGTA', 'AAAAAAAAAAAC', 'ACCGGGGGGGGT']

  orig3mer = ['TGG', 'CGG', 'GTG', 'GCG', 'GGT', 'GGC']
  orig12mer = ['GGGGGGGGGTGG', 'GGGGGGGGGCGG', 'GGGGGGGGGGTG', 'GGGGGGGGGGCG', 'GGGGGGGGGGGT', 'GGGGGGGGGGGC', 'GGGGGGGGTGGC', 'GGGGGGGGCGGC', 'GGGGGGGTCGGC', 'GGGGGGGCCGGC', 'GGGGGGTCCGGC', 'GGGGGGCCCGGC', 'GGGGGTCCCGGC', 'GGGGGCCCCGGC', 'GGGGTCCCCGGC', 'GGGGCCCCCGGC', 'GGGTCCCCCGGC', 'GGGCCCCCCGGC', 'GGTCCCCCCGGC', 'GGCCCCCCCGGC', 'GTCCCCCCCGGC', 'GCCCCCCCCGGC', 'TCCCCCCCCGGC', 'CCCCCCCCCGGC']
  print('original 3mer res set used, TGG, CGG, GTG, GCG, GGT, GGC', checkUniqueHamming(orig3mer, 3, 'ACGT')) #True
  print('original 12mer res set used '+','.join(orig12mer), checkUniqueHamming(orig12mer, 12, 'ACGT')) #True

#find and save all minimal resolving sets of DNA 3mers
def all3merResSets():
  size = 1
  resSets = []
  while len(resSets)==0:
    print('size', size)
    resSets = hammingBruteForce(3, 'ACGT', size)
    size += 1
  print(len(resSets), len(resSets[0]), resSets[0])
  np.save('RandResSets/minimal_3mer_resSets.npy', np.array(resSets))

if __name__=='__main__':
    findResSets()
#    all3merResSets()

