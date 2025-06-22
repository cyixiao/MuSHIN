import torch
import torch.nn as nn
import internal.utils.utils as utils


# Generate hyperedge features by aggregating node features based on the incidence matrix
def generate_hyperedge_features(node_features, incidence_matrix, aggregation="mean"):
    num_nodes, node_dim = node_features.shape
    num_edges = incidence_matrix.shape[1]

    if incidence_matrix.is_sparse:  # Handle sparse incidence matrices
        incidence_matrix = incidence_matrix.coalesce()
        if aggregation == "mean":
            degree = torch.sparse.sum(incidence_matrix, dim=0).to_dense()
            degree[degree == 0] = 1
        weighted_node_features = torch.sparse.mm(incidence_matrix.T, node_features)
        if aggregation == "mean":
            hyperedge_features = weighted_node_features / degree.unsqueeze(1)
        else:
            hyperedge_features = weighted_node_features
    else:  # Handle dense incidence matrices
        weighted_node_features = torch.mm(incidence_matrix.T, node_features)
        degree = torch.sum(incidence_matrix, dim=0, keepdim=True)
        degree[degree == 0] = 1
        if aggregation == "mean":
            hyperedge_features = weighted_node_features / degree.T
        else:
            hyperedge_features = weighted_node_features

    return hyperedge_features


# Node-to-edge attention mechanism for aggregating node features into hyperedge features
class NodeToEdgeAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):
        super(NodeToEdgeAttention, self).__init__()
        self.node_transform = nn.Linear(node_dim, hidden_dim)
        self.attention_weight = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.expand_output = nn.Linear(hidden_dim, output_dim)

    # Forward pass for node-to-edge attention
    def forward(self, node_features, incidence_matrix):
        num_edges = incidence_matrix.shape[1]
        transformed_node_features = self.node_transform(node_features)
        masked_node_features = torch.mm(incidence_matrix.T, transformed_node_features)
        attention_scores = self.leaky_relu(self.attention_weight(masked_node_features))
        attention_coeffs = torch.nn.functional.softmax(
            attention_scores.to_dense(), dim=0
        )
        weighted_edge_features = attention_coeffs * masked_node_features
        updated_edge_features = self.expand_output(weighted_edge_features)
        updated_edge_features = utils.min_max_normalize_cuda(updated_edge_features)
        return updated_edge_features


# Edge-to-node attention mechanism for aggregating hyperedge features back into node features
class EdgeToNodeAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim):
        super(EdgeToNodeAttention, self).__init__()
        self.node_transform = nn.Linear(node_dim, hidden_dim)
        self.edge_transform = nn.Linear(edge_dim, hidden_dim)
        self.attention_weight = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.expand_output = nn.Linear(hidden_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    # Forward pass for edge-to-node attention
    def forward(self, node_features, incidence_matrix, edge_features):
        num_nodes, node_dim = node_features.shape
        num_edges, edge_dim = edge_features.shape
        transformed_node_features = self.node_transform(node_features)
        transformed_edge_features = self.edge_transform(edge_features)
        aggregated_edge_features = torch.mm(
            incidence_matrix, transformed_edge_features
        ) / (incidence_matrix.sum(dim=1, keepdim=True) + 1e-8)
        attention_input = transformed_node_features + aggregated_edge_features
        attention_scores = self.leaky_relu(self.attention_weight(attention_input))
        attention_coeffs = torch.sigmoid(attention_scores)
        weighted_edge_features = attention_coeffs * aggregated_edge_features
        updated_node_features = weighted_edge_features + transformed_node_features
        updated_node_features = self.expand_output(updated_node_features)
        updated_node_features = utils.min_max_normalize_cuda(updated_node_features)
        return updated_node_features


# Multi-head node-to-edge attention for capturing diverse relationships between nodes and hyperedges
class MultiHeadNodeToEdgeAttention(nn.Module):
    def __init__(
        self, node_dim, edge_dim, hidden_dim, output_dim, num_heads=4, dropout_rate=0.3
    ):
        super(MultiHeadNodeToEdgeAttention, self).__init__()
        self.num_heads = num_heads
        self.attentions = nn.ModuleList(
            [
                NodeToEdgeAttention(node_dim, edge_dim, hidden_dim, edge_dim)
                for _ in range(num_heads)
            ]
        )
        self.relu = nn.ReLU()
        self.output_transform = nn.Linear(num_heads * edge_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    # Forward pass for multi-head node-to-edge attention
    def forward(self, node_features, incidence_matrix):
        multi_head_outputs = [
            attention(node_features, incidence_matrix) for attention in self.attentions
        ]
        concatenated_output = torch.cat(multi_head_outputs, dim=-1)
        concatenated_output = self.relu(concatenated_output)
        concatenated_output = self.dropout(concatenated_output)
        return self.output_transform(concatenated_output)


# Multi-head edge-to-node attention for capturing diverse relationships between hyperedges and nodes
class MultiHeadEdgeToNodeAttention(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim,
        num_heads=4,
        output_dim=128,
        dropout_rate=0.3,
    ):
        super(MultiHeadEdgeToNodeAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_heads = nn.ModuleList(
            [
                EdgeToNodeAttention(node_dim, edge_dim, hidden_dim, output_dim)
                for _ in range(num_heads)
            ]
        )
        self.relu = nn.ReLU()
        self.output_transform = nn.Linear(num_heads * output_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    # Forward pass for multi-head edge-to-node attention
    def forward(self, node_features, incidence_matrix, edge_features):
        multi_head_outputs = node_features
        for head in self.attention_heads:
            multi_head_outputs = head(
                multi_head_outputs, incidence_matrix, edge_features
            )
            multi_head_outputs = self.relu(multi_head_outputs)
        output = self.dropout(multi_head_outputs)
        return output
