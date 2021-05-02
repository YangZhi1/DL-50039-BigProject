import networkx as nx

import torch
import torch.nn as nn

import random

def read_data(edge_file, feature_file):
    '''
    Reads edge and feature files and returns a networkx Graph object containing the nodes, edges, and the nodes features.

    Parameters:
        edge_file (string): Path to the file containing the edges of the graph
        feature_file (string): Path to the file containing the features of each node of the graph

    Returns:
        A networkx Graph object built from the information provided in the two files
        A sorted list containing the keys of the features that all nodes have
    '''
    G = nx.Graph()
    features = set()

    # Add all the nodes and their corresponding features
    with open(feature_file, 'r') as ff:
        for line in ff.readlines():
            l = list(map(int, line.strip().split()))
            # the first element is the node
            G.add_node(l[0])
            for index, feature in enumerate(l[1:]):
                G.nodes[l[0]][index] = feature
                features.add(index)
    
    with open(edge_file, 'r') as ef:
        for line in ef.readlines():
            to_node, from_node = map(int, line.strip().split())
            G.add_edge(to_node, from_node)
    
    return G, sorted(list(features))


def generate_samples(G, features, num_scammers=1, defining_feature=19, probs=(0.9, 0.05), num_victims=1, max_hop=2, use_hop_distance=True, num_data_pts=100):
    '''
    Given a networkx Graph object, generate num_data_pts points of data.

    If `defining_feature` exists is given and is a valid feature, then `probs` would be used to select victims.
    If a node within `max_hop` distance away has the same `defining_feature` as a scammer, it would be labelled as a victim with probability `probs[0]`.
    Otherwise, it would be a victim with probability `probs[1]`.

    On the other hand, if `defining_feature` is `None`, then `num_victims` victims would be sampled randomly out of all nodes that are within `max_hop` distance away.

    Parameters:
        G (networkx Graph): A networkx Graph object containing the graph to extract the data from
        features (list of ints): Which features to include in the input data
        num_scammers (int): [Optional] How many scammers to place
        defining_feature (Object): [Optional] Which is the feature to be used as the main defining feature
        probs (list of float): [Optional] The chance that a node becomes a victim based on the main defining feature
        num_victims (int): [Optional] How many victims to place
        max_hop (int): [Optional] The maximum hop from the scammer to the victim (set this to None if anybody can be victim (regardless of hop distance))
        use_hop_distance (bool): [Optional] Whether or not to include hop distances from scammers as input features
        num_data_pts (int): [Optional] How many data points ((X, y) pairs) to generate
    
    Returns:
        The input data (X) is a [num_nodes * (1 + len(features))] list. The first sublist refers to whether that indexed node is the scammer (1 if it is).
        The output data (y) is a [num_nodes * 1] list, where all except one element are zeros. The index whose element is non-zero (1) is the victim.
        Returns [num_data_pts] pairs of (X, y) pairs.
    '''
    Xs = []
    ys = []

    for _ in range(num_data_pts):
        # choose a node to act as the scammer
        scammers = random.sample(list(G.nodes), min(num_scammers, len(G.nodes)))            

        # collect all the possible victims
        victims = set()
        for scammer in scammers:
            # get nodes which are certain hops away
            t_victims = nx.single_source_shortest_path_length(G, scammer, max_hop)
            
            # delete scammers from above output
            del t_victims[scammer]

            # add them to a set
            for victim in t_victims:
                victims.add(victim)
        
        # remove the scammers from the set of possible victims
        for scammer in scammers:
            if scammer in victims:
                victims.remove(scammer)

        # use the defining feature to generate victims
        if defining_feature is not None:
            victim = []
            for vic in victims:
                for scammer in scammers:
                    # if the features of the node match that of the scammer, we assume that it has a high chance of being targeted
                    if G.nodes[vic].get(defining_feature, 0) == G.nodes[scammer].get(defining_feature, 0):
                        if random.random() < probs[0]:
                            victim.append(vic)
                    else:
                        if random.random() < probs[1]:
                            victim.append(vic)
        else:
            # randomly choose some victims
            victim = random.sample(list(victims), min(num_victims, len(victims)))

        # create X and y lists
        X = [[1] if node in scammers else [0] for node in G.nodes]
        y = [1 if node in victim else 0 for node in G.nodes]

        # extract hop distances
        if use_hop_distance:
            distances = dict(nx.all_pairs_shortest_path_length(G))
            for i, node in enumerate(G.nodes):
                for scammer in scammers:
                    X[i].append(distances[scammer][node])

        # extract the features from the nodes
        for i, node in enumerate(G.nodes):
            for feature_index in features:
                try:
                    feat = G.nodes[node][feature_index]
                except KeyError:
                    feat = 0
                X[i].append(feat)
            
        # append that to respective lists
        Xs.append(X)
        ys.append(y)

    return (Xs, ys)

class GCNKipf_Layer(nn.Module):
    '''
    Kipf GCN convolution layer class.
    '''
    def __init__(self, adj, input_channels, output_channels, device):
        super().__init__()
        self.A_hat = adj + torch.eye(adj.size(0)).to(device)
        self.D = torch.diag(torch.sum(adj, 1))
        self.D = self.D.inverse().sqrt()
        self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)
        self.W = nn.Parameter(torch.rand(input_channels, output_channels))
    
    def forward(self, X):
        out = torch.relu(torch.mm(torch.mm(self.A_hat, X), self.W))
        return out

class Net(nn.Module):
    '''
    Standard GCN model class.
    '''
    def __init__(self, adj, num_feat, num_hid, num_out, device):
        super().__init__()
        self.conv1 = GCNKipf_Layer(adj, num_feat, num_hid, device).to(device)
        self.conv2 = GCNKipf_Layer(adj, num_hid, num_out, device).to(device)
        
    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        return X