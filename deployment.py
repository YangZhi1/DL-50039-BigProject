# required libraries
import networkx as nx
import matplotlib.pyplot as plt
import torch

# project-specific functions
import graph
from dimcli.utils.networkviz import *

def get_graph(node_index1, node_index2):
    plt.clf()
    # read data and build graph object
    try:
      G, _ = graph.read_data('3980_edited.edges', '3980_edited.feat')
    except FileNotFoundError:
      G, _ = graph.read_data('data/3980_edited.edges', 'data/3980_edited.feat')
    # check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)
    
    scammer_select = (int(node_index1), int(node_index2))
    try:
      model_path = "model_final.pt"
    except FileNotFoundError:
      model_path = "models/model_final.pt" 
    X = []
    distances = dict(nx.all_pairs_shortest_path_length(G))
    for node in G.nodes:
        in_feat = []
        # scammer label
        if node in scammer_select:
            in_feat.append(1)
        else:
            in_feat.append(0)
        # hop distance to scammer 1
        in_feat.append(distances[scammer_select[0]][node])
        # hop distance to scammer 2
        in_feat.append(distances[scammer_select[1]][node])
        # feature 19
        in_feat.append(G.nodes[node][19])
        X.append(in_feat)

    # load adjacency matrix
    A = nx.adjacency_matrix(G).todense()
    A = torch.Tensor(A).to(device)

    # get the input feature size
    input_size = len(X[0])

    # load the model
    model = graph.Net(A, input_size, 10, 2, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # run this through our model
    X = torch.Tensor(X).to(device)
    out = torch.argmax(model(X), dim=1)
    
    node_color = out
    grey = "#C0C0C0"
    gold = "#FFD700"
    colours = []
    for i in out:
      if i == 1:
        colours.append(grey)
      else:
        colours.append(gold)
        
    nx.draw_networkx(G, node_color=colours, font_size=8)
    plt.savefig("graphimage.png")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    