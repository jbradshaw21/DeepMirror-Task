from tdc.single_pred import ADME
from pysmiles import read_smiles
import networkx as nx
import logging

import torch
from torch.nn import Dropout
from torch_geometric.nn import MessagePassing, global_max_pool, GCNConv

import time
import numpy as np

# Prevents warning messages clogging output (see 'Atom "[...]" contains sterochemical information...' in README)
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)  # Anything higher than warning

# Initalise params
start_time = time.time()
learning_rate = 1e-3
weight_decay = 0
dropout_rate = 0.6
stopping_criteria = 1e-5
batch_size = 16
test = False    # test if True valid if False
dataset_name = 'Half_Life_Obach'    # trade for other datasets (all use the same features)

np.random.seed(42)
torch.manual_seed(42)

data = ADME(name=dataset_name)
split = data.get_split()

train_loader, train_labels, valid_loader, valid_labels, test_loader, test_labels = [None for _ in range(6)]
for data_set in ['train', 'valid', 'test']:
    mol_count = 0
    batch, batch_labels, batch_loader, label_loader = [[] for _ in range(4)]    # we will load data into batches...
    for drug_idx, smiles_string in enumerate(split[data_set]['Drug'], 0):
        mol = read_smiles(smiles_string)    # using PySmiles to extract graph data from 'smiles string'
        batch.append(mol)
        batch_labels.append(split[data_set]['Y'][drug_idx])
        mol_count += 1
        if mol_count % batch_size == 0 or mol_count == len(split[data_set]):    # allocate 'full' batches of data
            batch_loader.append(batch)
            label_loader.append(batch_labels)
            batch, batch_labels = [[] for _ in range(2)]

    # designate to appropriate dataset
    if data_set == 'train':
        train_loader = batch_loader
        train_labels = label_loader
    elif data_set == 'valid':
        valid_loader = batch_loader
        valid_labels = label_loader
    elif data_set == 'test':
        test_loader = batch_loader
        test_labels = label_loader


# simple GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16, add_self_loops=False)
        self.conv2 = GCNConv(16, 1, add_self_loops=False)

        self.dropout = Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


# initialise model properties
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.L1Loss().to(device)


def train(model, data_loader, labels, optimizer):
    model.train()
    total_loss = 0
    for data_idx, _ in enumerate(data_loader, 0):
        optimizer.zero_grad()

        # extract the edges from each graph in the batch
        graph_edges = [nx.to_pandas_edgelist(graph) for graph in data_loader[data_idx]]
        edge_index = [torch.tensor(edge_list[['source', 'target']].to_numpy()) for edge_list in graph_edges]

        # edge index must be cumulative, so shift indices accordingly
        shift = [torch.max(tens) for tens in edge_index]
        pos_shift = np.cumsum([len(tens) for tens in edge_index])
        edge_index = torch.cat(edge_index)

        for shift_idx in range(len(shift)):
            edge_index[pos_shift[shift_idx]:] += shift[shift_idx]

        # torch.reshape produces the wrong output, so I hardcode this functionality
        reshape_edge = torch.zeros(size=(edge_index.shape[1], edge_index.shape[0]))
        for edge_idx, edge in enumerate(edge_index, 0):
            reshape_edge[0][edge_idx] = edge[0]
            reshape_edge[1][edge_idx] = edge[1]
        edge_index = reshape_edge

        # use node degrees as 'node features' (inappropriate?)
        graph_degree = [graph.degree for graph in data_loader[data_idx]]
        degree_list = [val for (_, val) in graph_degree[0]]
        for graph_idx in range(1, len(graph_degree)):
            degree_list += [val for (_, val) in graph_degree[graph_idx]]
        degree_list = torch.reshape(torch.tensor(degree_list), shape=(len(degree_list), 1))

        # attempt to convert node logits to graph logits...
        num_nodes = [max(graph.nodes) for graph in data_loader[data_idx]]
        node_logits = model(degree_list.float().to(device), edge_index.type(torch.LongTensor).to(device))
        graph_logits = [node_logits[np.cumsum(num_nodes)[node_idx]:np.cumsum(num_nodes)[node_idx + 1]] for node_idx
                        in range(len(num_nodes) - 1)]

        # compute the loss
        logits = torch.stack([torch.mean(graph, dim=0) for graph in graph_logits])
        loss = criterion(logits, torch.tensor(labels[data_idx]).type(torch.LongTensor).to(device))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def test(model, data_loader, labels):
    model.eval()

    total_loss = 0
    for data_idx, _ in enumerate(data_loader, 0):
        torch.no_grad()

        # extract the edges from each graph in the batch
        graph_edges = [nx.to_pandas_edgelist(graph) for graph in data_loader[data_idx]]
        edge_index = [torch.tensor(edge_list[['source', 'target']].to_numpy()) for edge_list in graph_edges]

        # edge index must be cumulative, so shift indices accordingly
        shift = [torch.max(tens) for tens in edge_index]
        pos_shift = np.cumsum([len(tens) for tens in edge_index])
        edge_index = torch.cat(edge_index)

        for shift_idx in range(len(shift)):
            edge_index[pos_shift[shift_idx]:] += shift[shift_idx]

        # torch.reshape produces the wrong output, so I hardcode this functionality
        reshape_edge = torch.zeros(size=(edge_index.shape[1], edge_index.shape[0]))
        for edge_idx, edge in enumerate(edge_index, 0):
            reshape_edge[0][edge_idx] = edge[0]
            reshape_edge[1][edge_idx] = edge[1]
        edge_index = reshape_edge

        # use node degrees as 'node features' (inappropriate?)
        graph_degree = [graph.degree for graph in data_loader[data_idx]]
        degree_list = [val for (_, val) in graph_degree[0]]
        for graph_idx in range(1, len(graph_degree)):
            degree_list += [val for (_, val) in graph_degree[graph_idx]]
        degree_list = torch.reshape(torch.tensor(degree_list), shape=(len(degree_list), 1))

        # attempt to convert node logits to graph logits...
        num_nodes = [max(graph.nodes) for graph in data_loader[data_idx]]
        node_logits = model(degree_list.float().to(device), edge_index.type(torch.LongTensor).to(device))
        graph_logits = [node_logits[np.cumsum(num_nodes)[node_idx]:np.cumsum(num_nodes)[node_idx + 1]] for node_idx
                        in range(len(num_nodes) - 1)]

        # compute the loss
        logits = torch.stack([torch.mean(graph, dim=0) for graph in graph_logits])
        loss = criterion(logits, torch.tensor(labels[data_idx]).type(torch.LongTensor).to(device))

        total_loss += loss.item()

    return total_loss


# chooses either valid/test set based on selection
if not test:
    test_loader = valid_loader
    test_labels = valid_labels

# initialise some variables that are useful for tracking training...
test_loss, converged, prev_loss, epoch = [None, False, [], 0]
while not converged:
    train_loss = train(model, train_loader, train_labels, optimizer)
    test_loss = test(model, test_loader, test_labels)
    print(f'Epoch: {epoch + 1}, Train Loss: {round(train_loss, 4)}, Test Loss: {round(test_loss, 4)}'
          f' Time: {round(time.time() - start_time, 2)} seconds')

    epoch += 1
    prev_loss.append(train_loss)
    if len(prev_loss) >= 5:
        std = np.std(prev_loss[-5:])    # converge when std of training_loss falls below a given threshold
        if std < stopping_criteria:
            converged = True

print(f'Final metrics after {epoch} epochs: Test Loss: {round(test_loss, 4)},'
      f' Time: {round(time.time() - start_time, 4)} seconds')
