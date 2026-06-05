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
    elif args.graph_model == 'ArmoredHub':
        # Ensure it is an undirected graph so get_graph_info works properly!
        graph = nx.Graph()
        graph.add_nodes_from(range(args.nodes))

        # Define how big each lobe/cave is
        cave_size = 8  
        
        # Identify the Hubs (Node 0, 8, 16, 24...)
        hubs = [i for i in range(args.nodes) if i % cave_size == 0]

        # 1. Build the Isolated Caves
        for i in range(args.nodes):
            my_cave_start = (i // cave_size) * cave_size
            my_hub = my_cave_start
            
            if i not in hubs:
                # RULE A: Local nodes MUST connect to their Hub
                graph.add_edge(i, my_hub)
                
                # RULE B: Local nodes connect to their local neighbors
                for neighbor in range(my_cave_start, my_cave_start + cave_size):
                    if i != neighbor: 
                        graph.add_edge(i, neighbor)

        # 2. Build the Hub-to-Hub Super-Highway
        # We create a small Small-World (WS) graph just for the hubs
        # This prevents bottlenecks between distant lobes
        hub_subgraph = nx.random_graphs.connected_watts_strogatz_graph(len(hubs), k=2, p=0.5)
        
        # Map the sub-graph edges back to the main graph's Hub IDs
        for u, v in hub_subgraph.edges():
            graph.add_edge(hubs[u], hubs[v])

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
