from collections import defaultdict
import numpy as np
import random
import networkx as nx
from itertools import permutations
from itertools import combinations
import time
from line_profiler import LineProfiler
lp = LineProfiler()

WEIGHT_LIMIT = 1000 #Default according to GSP
NUM_NODES = [3,4,5,6,7,10,15] # list used to generate test cases with specified number of nodes. Feel free to modify.

def generate_tsp_test_case(num_nodes, weight_limit):
    "Generate a complete graph with random weights"
    G = nx.complete_graph(num_nodes)

    for u,v in G.edges():
        G[u][v]['weight'] = random.randint(1, weight_limit)

    return G

def traveling_salesman_brute_force(G, start_node = 0):
    "Computes the cheapest TSP tour by brute force"
    dp = {}
    keys = []
    nodes = set(G.nodes)
    nodes.remove(start_node)

    d = lambda u, v : G[u][v]['weight']
    # create keys
    for k in range(len(nodes) + 1):
      for subset in subsets_of_size_k(nodes, k):
        keys.append(frozenset(subset))
        dp[frozenset(subset)] = {i: float('inf') for i in nodes}

    # base case
    for i in nodes:
      dp[keys[0]][i] = d(i, start_node)

    for j in range(1, len(keys)):
      k = keys[j]
      S = set(k)
      for i in nodes:
        if i in S:
          continue
        for t in S:
          sub = S - set([t])
          dp[k][i] = min(dp[k][i], dp[frozenset(sub)][t] + d(i, t))

    return [], min(dp[frozenset(nodes - set([i]))][i] + d(i, start_node) for i in nodes)

def tsp_solver(G, start_vertex):
  N = len(G.nodes)

  nodes = list(G.nodes)
  nodes.remove(start_vertex)
  np_nodes = np.array(G.nodes)
  np_nodes = (1 << np_nodes)

  dp = np.full((2 ** N, N), np.inf)

  d = np.zeros((N, N))
  for u in G.nodes:
    for v in G.nodes:
      if u != v:
        d[u, v] = G[u][v]['weight']

  # base case
  dp[0] = d[start_vertex]

  max_w = d.sum().sum()
  in_mask = np.zeros(dp.shape, dtype=bool)
  keys = np.arange(2 ** N)

  for n in G.nodes:
    in_mask[:, n] = keys & np_nodes[n] > 0


  for mask in range(1, 2 ** N):
    if mask & (1 << start_vertex):
      continue
    # sub_w = np.diag(dp[mask & ~np_nodes]) # (1, N)
    dp[mask] = (d + in_mask[mask] * max_w + np.diag(dp[mask & ~np_nodes])[:, None])[in_mask[mask]].min(0)

  mask_no_start_v = (2 ** N - 1) & ~(1 << start_vertex)

  return min(dp[mask_no_start_v & ~(1 << i)][i] + d[i, start_vertex] for i in nodes)


def runTest():
    num_correct = 0
    for i in NUM_NODES:
        test_g = generate_tsp_test_case(i,WEIGHT_LIMIT)
        print(f"Testing randomly generated graph with {i} nodes")
        #Brute force to get solutions to compare against. Also measure time
        b_start = time.time()
        b_tour, b_weight = traveling_salesman_brute_force(test_g)
        b_end = time.time()
        b_time = b_end - b_start
        print(f"Brute Force solution weight: {b_weight}, Time: {b_time}")

        #Test implementation
        dp_start = time.time()
        dp_weight = tsp_solver(test_g,0) #Default start at 0 for this test to follow bruteforce soln
        dp_end = time.time()
        dp_time = dp_end - dp_start
        print(f"DP solution: {dp_weight}, Time: {dp_time}")

        if b_weight == dp_weight:
            print("Correct soln!\n\n")
            num_correct+=1
        else:
            print("NOT correct :(\n\n")
    print(f"Total correct solns: {num_correct} out of {len(NUM_NODES)}")



if __name__ == "__main__":
    lp_wrapper = lp(tsp_solver)
    lp_wrapper(generate_tsp_test_case(15, WEIGHT_LIMIT), 0)
    lp.print_stats()

    # python -m line_profiler test.py.lprof > profile_output.txt