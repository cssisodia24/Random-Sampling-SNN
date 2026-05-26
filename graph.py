import networkx as nx
import yaml
import collections
import numpy as np
import random


Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])
def distance_biased_watts_strogatz(n, k, p, decay_factor=0.5, seed=None):
    """
    A modified Watts-Strogatz generator tailored for Spiking Neural Networks.
    Introduces a vision-prior by biasing rewiring targets to local neighborhoods.
    
    n: number of nodes (e.g., 32)
    k: initial nearest neighbors (e.g., 4)
    p: rewiring probability (e.g., 0.75)
    decay_factor: controls how strictly local the rewiring is. 
                  High = strict local clusters. 0 = standard uniform random WS.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # 1. Build the initial Ring Lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            target = (i + j) % n
            G.add_edge(i, target)

    # 2. Distance-Biased Rewiring Phase
    nodes = list(G.nodes())
    
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if random.random() < p:
                original_target = (i + j) % n
                
                valid_targets = []
                probabilities = []
                
                # Evaluate every node in the graph as a potential target
                for target in nodes:
                    # Enforce basic rules: No self-loops, no duplicate edges
                    if target == i or G.has_edge(i, target):
                        continue
                        
                    # Calculate the shortest physical distance on the ring
                    distance = min(abs(i - target), n - abs(i - target))
                    
                    # Apply exponential decay: e^(-decay_factor * distance)
                    weight = np.exp(-decay_factor * distance)
                    
                    valid_targets.append(target)
                    probabilities.append(weight)
                
                # Normalize the weights so they sum exactly to 1.0 (creating a probability distribution)
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                
                # Pick the new target using the distance-biased probability distribution
                new_target = np.random.choice(valid_targets, p=probabilities)
                
                # Execute the rewiring
                G.remove_edge(i, original_target)
                G.add_edge(i, new_target)

    return G


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
        # Now using the Distance-Biased Small World model
        # You can adjust the decay_factor here to test different levels of spatial locality
        return distance_biased_watts_strogatz(Nodes, args.K, args.P, decay_factor=0.5, seed=args.graph_seed)
    elif args.graph_model == 'GNM':
        return nx.random_graphs.gnm_random_graph(Nodes, args.M)

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
