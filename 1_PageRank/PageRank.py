links = {}
fnames = ['angelinajolie.html', 'bradpitt.html',
        'jenniferaniston.html', 'jonvoight.html',
        'martinscorcese.html', 'robertdeniro.html']

for files in fnames:
    links[files] = []
    f = open(files, 'r')
    for line in f.readlines():
        while True:
            p = line.partition('<a href="http://')[2]
            if p == '':
                break
            url,_,line = p.partition('\">')
            links[files].append(url)
    f.close()

import networkx as nx
DG = nx.DiGraph()
DG.add_nodes_from(fnames)
edges = []
for key, values in links.iteritems():
    eweight = {}
    for v in values:
        if v in eweight:
            eweight[v] += 1
        else:
            eweight[v] = 1
    for succ, weight in eweight.iteritems():
        edges.append([key, succ, {'weight': weight}])

DG.add_edges_from(edges)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,9))
pos=nx.spring_layout(DG, iterations=10)
nx.draw(DG, pos, node_size = 0, alpha = 0.4, edge_color = 'r', font_size = 16)
# plt.savefig('link_graph.png')
plt.show()

from numpy import matrix, zeros
NX = len(fnames)
T = matrix(zeros((NX, NX)))

f2i = dict((fn, i) for i, fn in enumerate(fnames))

for predecessor, successor in DG.adj.iteritems():
    for s, edata in successor.iteritems():
        T[f2i[predecessor], f2i[s]] = edata['weight']

from numpy.random import random
from numpy import sum, ones, dot
epsilon = 0.01
E = ones(T.shape)/NX
L = T + epsilon * E
G = matrix(zeros(L.shape))
for i in xrange(NX):
    G[i,:] = L[i,:] / sum(L[i,:])

PI = random(NX)
PI /= sum(PI)
R = PI
for _ in xrange(100):
    R = dot(R, G)

evolution = [dot(PI, G**i) for i in xrange(1, 20)]

plt.figure()
for i in xrange(NX):
    plt.plot([step[0,i] for step in evolution], label=fnames[i], lw=2)

plt.title('rank vs iterations')
plt.xlabel('iterations')
plt.ylabel('rank')
plt.legend()
plt.draw()
# plt.savefig('rank_vs_iteration.png')
plt.show()

revind = {}
for fname in fnames:
    for line in open(fname).readlines():
        for token in line.split():
            if token in revind:
                if fname in revind[token]:
                    revind[token][fname] += 1
                else:
                    revind[token][fname] = 1
            else:
                revind[token] = {fname: 1}

def getPageRank(fname):
    return R[0, f2i[fname]]

result = revind['film'].keys()
result = sorted(result, key=getPageRank, reverse=True)
print result
