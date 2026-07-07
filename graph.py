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
    elif args.graph_model == 'SpinGlass':
        import numpy as np
        import math
        
        N = int(args.nodes)
        graph = nx.Graph()
        graph.add_nodes_from(range(N))
        
        # 1. FAIRNESS: Define edge budget (2 * N to match baseline)
        edge_budget = N * 2
        
        # 2. Random spin initialization
        spins = np.random.choice([-1, 1], size=N)
        
        # 3. Interaction strength matrix
        # J_ij represents the coupling strength between nodes i and j
        J = np.random.normal(0, 1, (N, N))
        
        # 4. Energy Contribution Matrix
        # We calculate the strength of interaction: H_ij = J_ij * s_i * s_j
        # We want to connect nodes that contribute most to minimizing (or stabilizing) 
        # the system energy.
        interaction_matrix = J * np.outer(spins, spins)
        
        # Ensure the interaction is symmetric for an undirected graph
        interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
        
        # Set diagonal to -inf so we don't connect nodes to themselves
        np.fill_diagonal(interaction_matrix, -np.inf)
        
        # 5. RANK-BASED SELECTION
        # We want the strongest coupling interactions (highest absolute value)
        # Sort indices by interaction strength in descending order
        flat_indices = np.argsort(np.abs(interaction_matrix.ravel()))[::-1]
        
        edges_added = 0
        for idx in flat_indices:
            if edges_added >= edge_budget:
                break
            
            u, v = divmod(idx, N)
            
            # Ensure unique, valid undirected edges
            if u < v and not graph.has_edge(u, v):
                graph.add_edge(u, v)
                edges_added += 1
                    
        return graph
    elif args.graph_model == 'SpatialSpinGlass':
        import networkx as nx
        import numpy as np
        import math
        
        N = int(args.nodes)
        graph = nx.Graph()
        graph.add_nodes_from(range(N))
        
        # 1. FAIRNESS: Define edge budget (2 * N to match baseline)
        edge_budget = N * 2
        
        # 2. PHYSICAL GEOMETRY (The missing ingredient from pure Spin Glass)
        side_length = int(math.ceil(math.sqrt(N)))
        coords = np.array([(i // side_length, i % side_length) for i in range(N)])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        
        # 3. Random spin initialization
        spins = np.random.choice([-1, 1], size=N)
        
        # 4. SPATIALLY-BIASED Interaction Matrix
        # We generate random coupling, but divide it by physical distance.
        # This means local nodes naturally have stronger coupling limits than distant ones.
        J_random = np.random.normal(0, 1, (N, N))
        J = J_random / (distances + 1.0)  # +1.0 prevents division by zero
        
        # 5. Energy Contribution Matrix: H_ij = J_ij * s_i * s_j
        interaction_matrix = J * np.outer(spins, spins)
        interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
        
        # Set diagonal to -inf so we don't connect nodes to themselves
        np.fill_diagonal(interaction_matrix, -np.inf)
        
        # 6. RANK-BASED SELECTION
        flat_indices = np.argsort(np.abs(interaction_matrix.ravel()))[::-1]
        
        edges_added = 0
        for idx in flat_indices:
            if edges_added >= edge_budget:
                break
            
            u, v = divmod(idx, N)
            
            if u < v and not graph.has_edge(u, v):
                graph.add_edge(u, v)
                edges_added += 1
                    
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
