import networkx as nx
import yaml
import collections
import numpy
import random


Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])


def get_graph_info(graph):
    input_nodes = []
    output_nodes = []
    Nodes = []
    for node in range(graph.number_of_nodes()):
        tmp = list(graph.neighbors(node))
        tmp.sort()
        type = -1
        if len(tmp) == 0:
            input_nodes.append(node)
            output_nodes.append(node)
            type = 0
        else:
            if node < tmp[0]:
                input_nodes.append(node)
                type = 0
            if node > tmp[-1]:
                output_nodes.append(node)
                type = 1
        Nodes.append(Node(node, [n for n in tmp if n < node], type))
    return Nodes, input_nodes, output_nodes


# randomly replace edge based on graph
def get_skip_graph(nodes, input_nodes, output_nodes, skip_ratio):
    skip_graph = []
    for id, node in enumerate(nodes):
        input_id = []
        for _id in node.inputs:
            if random.random() <= skip_ratio and len(input_id) < len(node.inputs) - 1:
                input_id.append(_id)
                # print(_id, id)
        skip_graph.append(input_id)
        for _id in input_id:
            node.inputs.remove(_id)
    return skip_graph


def build_graph(Nodes, args):
    args.graph_seed += 1
    if args.graph_model == 'ER':
        return nx.random_graphs.erdos_renyi_graph(Nodes, args.P, args.graph_seed)
    elif args.graph_model == 'BA':
        return nx.random_graphs.barabasi_albert_graph(Nodes, args.M, args.graph_seed)
    elif args.graph_model == 'WS':
        return nx.random_graphs.connected_watts_strogatz_graph(Nodes, args.K, args.P, tries=200, seed=args.graph_seed)
    elif args.graph_model == 'GNM':
        return nx.random_graphs.gnm_random_graph(Nodes, args.M)
    elif args.graph_model == 'ModularResonance':
        import networkx as nx
        import numpy as np

        graph = nx.Graph()
        N = int(args.nodes)
        # We use 1-based indexing for the modulo math to work perfectly
        nodes = list(range(1, N + 1))
        graph.add_nodes_from(nodes)

        # 1. The Local Feature Grid
        # Create a standard 2D lattice to process the image pixels
        side = int(np.sqrt(N))
        for idx in range(N):
            r, c = idx // side, idx % side
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side:
                    n_idx = nr * side + nc
                    # Add 1 to match our 1-based indexing
                    graph.add_edge(nodes[idx], nodes[n_idx])

        # 2. Find the next Prime Number P > N
        def is_prime(n):
            if n <= 1: return False
            for i in range(2, int(np.sqrt(n)) + 1):
                if n % i == 0: return False
            return True

        P = N + 1
        while not is_prime(P):
            P += 1

        # 3. The Cryptographic Shortcuts (Modular Multiplicative Inverse)
        for i in nodes:
            try:
                # Find j such that (i * j) mod P == 1
                j = pow(i, -1, P)
                # Ensure the target node actually exists in our network
                # and avoid self-loops
                if j <= N and i != j:
                    graph.add_edge(i, j)
            except ValueError:
                # If no inverse exists (rare with primes), just skip
                continue

        # Relabel nodes back to 0-based indexing to keep PyTorch happy
        mapping = {i: i - 1 for i in nodes}
        graph = nx.relabel_nodes(graph, mapping)

        return graph
    elif args.graph_model == 'MGGExpander':
        import networkx as nx
        import numpy as np
        import math

        graph = nx.Graph()
        N = int(args.nodes)
        graph.add_nodes_from(range(N))

        # MGG Expanders are built on 2D square fields
        m = int(math.ceil(math.sqrt(N)))

        def get_id(x, y):
            # Wrap around coordinates and map back to 1D node ID
            return (x % m) * m + (y % m)

        for x in range(m):
            for y in range(m):
                u = get_id(x, y)
                
                # If we exceed N due to rounding, skip (safe fallback for non-squares)
                if u >= N: 
                    continue

                # 1. Local Visual Grid (Feature Extraction)
                # Connect to Right and Down neighbors
                v1 = get_id(x + 1, y)
                v2 = get_id(x, y + 1)
                
                # 2. Algebraic Affine Shortcuts (Global Expressway)
                # Formula: (x, x+y) and (x+y, y) mod m
                v3 = get_id(x, x + y)
                v4 = get_id(x + y, y)

                # Add edges carefully to avoid self-loops and out-of-bounds
                for v in [v1, v2, v3, v4]:
                    if u != v and v < N:
                        graph.add_edge(u, v)

        return graph

def save_graph(graph, path):
    with open(path, 'w') as f:
        yaml.dump(graph, f)


def load_graph(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.Loader)


def calc_path(graph):
    nodes, input_nodes, output_nodes = get_graph_info(graph)
    num_path = {}
    len_path = {}
    num = 0
    len = 0
    for id, node in enumerate(nodes):
        if id in input_nodes:
            num_path[id] = 1
            len_path[id] = 1
        else:
            num_path[id] = 0
            len_path[id] = 0
            for _id in node.inputs:
                print(_id, id)
                num_path[id] += num_path[_id]
                len_path[id] += len_path[_id] + num_path[_id]
        print(id, num_path[id], len_path[id])
        if id in output_nodes:
            num += num_path[id]
            len += num_path[id] + len_path[id]

    return num, len
