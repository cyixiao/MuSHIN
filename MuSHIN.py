import torch
import torch.nn as nn
import torch_geometric.nn as hnn
from attentions import MultiHeadNodeToEdgeAttention, MultiHeadEdgeToNodeAttention

# MuSHIN: A multi-scale hypergraph neural network with optional attention mechanisms
class MuSHIN(nn.Module):
    def __init__(
        self,
        input_num,
        input_feature_num,
        emb_dim,
        conv_dim,
        head=3,
        p=0.1,
        L=1,
        use_attention=True,
        extra_feature=None,
        reaction_feature=None,
        enable_hygnn=False,
        incidence_matrix_pos=None,
    ):
        super(MuSHIN, self).__init__()

        # Initialize model parameters and layers
        self.emb_dim = emb_dim
        self.conv_dim = conv_dim
        self.p = p
        self.input_num = input_num
        self.head = head
        self.hyper_conv_L = L
        self.linear_encoder = nn.Linear(input_feature_num, emb_dim)
        self.similarity_liner = nn.Linear(input_num, emb_dim)
        self.max_pool = hnn.global_max_pool
        self.extra_feature = extra_feature
        self.incidence_matrix_pos = incidence_matrix_pos

        # Configure input channels based on additional features
        self.in_channel = emb_dim
        if extra_feature is not None:
            self.extra_feature = extra_feature
            self.in_channel = 2 * emb_dim

            # Handle reaction feature if provided
            self.reaction_feature = reaction_feature
            if reaction_feature is not None:
                self.linear_reaction_feature = nn.Linear(
                    self.reaction_feature.shape[1], emb_dim
                )

            self.enable_hygnn = enable_hygnn

            # Linear layer for extra features
            self.pre_linear = nn.Linear(extra_feature.shape[1], emb_dim)

            # Initialize attention mechanisms if enabled
            if self.enable_hygnn:
                self.linear_node_to_edge = nn.Linear(emb_dim, self.in_channel)
                self.node_to_edge_attr = MultiHeadNodeToEdgeAttention(
                    edge_dim=emb_dim,
                    node_dim=emb_dim,
                    hidden_dim=emb_dim * 2,
                    output_dim=self.in_channel,
                    num_heads=2,
                )
                self.edge_to_node_attr = MultiHeadEdgeToNodeAttention(
                    node_dim=emb_dim,
                    edge_dim=emb_dim * 2,
                    hidden_dim=emb_dim * 2,
                    output_dim=emb_dim,
                    num_heads=2,
                )

        # Activation function
        self.relu = nn.ReLU()

        # Initialize hypergraph convolution layers
        self.hypergraph_conv = hnn.HypergraphConv(
            self.in_channel,
            conv_dim,
            heads=head,
            use_attention=use_attention,
            dropout=p,
        )
        if L > 1:
            self.hypergraph_conv_list = nn.ModuleList()
            for l in range(L - 1):
                self.hypergraph_conv_list.append(
                    hnn.HypergraphConv(
                        head * conv_dim,
                        conv_dim,
                        heads=head,
                        use_attention=use_attention,
                        dropout=p,
                    )
                )

        # Initialize attention layers for hyperedges if enabled
        if use_attention:
            self.hyper_attr_liner = nn.Linear(input_num, self.in_channel)
            if L > 1:
                self.hyperedge_attr_list = nn.ModuleList()
                for l in range(L - 1):
                    self.hyperedge_attr_list.append(
                        nn.Linear(input_num, head * conv_dim)
                    )

        # Final linear layer for hyperedge classification
        self.hyperedge_linear = nn.Linear(conv_dim * head, 2)

        # Softmax for prediction
        self.softmax = nn.Softmax(dim=1)

    # Forward pass for MuSHIN
    def forward(self, input_features, incidence_matrix):
        # Prepare input features and incidence matrix
        incidence_matrix_T = incidence_matrix.T
        input_nodes_features = self.relu(self.linear_encoder(input_features))
        row, col = torch.where(incidence_matrix_T)
        edges = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
        hyperedge_attr = self.hyper_attr_liner(incidence_matrix_T)

        # Process extra features if provided
        if self.extra_feature is not None:
            extra_feature = self.pre_linear(self.extra_feature)
            if self.enable_hygnn and self.incidence_matrix_pos is not None:
                if self.reaction_feature is None:
                    updated_node_features = self.edge_to_node_attr(
                        extra_feature, self.incidence_matrix_pos, updated_edge_features
                    )
                    extra_feature = extra_feature + updated_node_features
                    hyperedge_attr = updated_edge_features
                else:
                    edge_feature = self.linear_reaction_feature(self.reaction_feature)
                    edge_feature = self.linear_node_to_edge(edge_feature)
                    updated_node_features = self.edge_to_node_attr(
                        extra_feature, self.incidence_matrix_pos, edge_feature
                    )
                    updated_edge_features = self.node_to_edge_attr(
                        updated_node_features, self.incidence_matrix_pos
                    )
                    updated_node_features = self.edge_to_node_attr(
                        extra_feature, self.incidence_matrix_pos, updated_edge_features
                    )
                    extra_feature = extra_feature + updated_node_features
                    hyperedge_attr = updated_edge_features

            extra_feature = self.relu(extra_feature)
            input_nodes_features = torch.cat(
                (extra_feature, input_nodes_features), dim=1
            )

        # Apply hypergraph convolution layers
        input_nodes_features = self.hypergraph_conv(
            input_nodes_features, edges, hyperedge_attr=hyperedge_attr
        )
        if self.hyper_conv_L > 1:
            for l in range(self.hyper_conv_L - 1):
                layer_hyperedge_attr = self.hyperedge_attr_list[l](incidence_matrix_T)
                input_nodes_features = self.hypergraph_conv_list[l](
                    input_nodes_features, edges, hyperedge_attr=layer_hyperedge_attr
                )
                input_nodes_features = self.relu(input_nodes_features)

        # Compute hyperedge features and return output
        hyperedge_feature = torch.mm(incidence_matrix_T, input_nodes_features)
        return self.hyperedge_linear(hyperedge_feature)

    # Predict function with softmax output
    def predict(self, input_fetures, incidence_matrix):
        return self.softmax(self.forward(input_fetures, incidence_matrix))
