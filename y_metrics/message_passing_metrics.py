import numpy as np

def custom_message_passing_metric(G, num_layers=6, decay_factor=0.7, feature_dim=50):
    """
    Computes a graph-level metric using message passing that:
    - Is learnable by GNNs (but requires multiple epochs)
    - Is sensitive to graph corruption
    - Prioritizes closer neighbors using decay
    - Avoids excessive normalization to maintain sensitivity to missing edges

    Args:
        G (nx.Graph): Input graph with node features in G.nodes[node]['x']
        num_layers (int): Number of message passing iterations
        decay_factor (float): Factor to reduce influence of distant neighbors (0-1)
        feature_dim (int): Dimension of node features

    Returns:
        float: A graph-level metric value
    """

    # Initialize node states with their features
    node_states = {}
    for node in G.nodes():
        # Initial state is the node feature
        node_states[node] = G.nodes[node]['x'].copy()

    # Track changes across layers (for final metric calculation)
    layer_changes = []

    # Message passing iterations
    for layer in range(num_layers):
        # Store the previous states for change calculation
        prev_states = {node: state.copy() for node, state in node_states.items()}

        # Message creation phase with custom functions varying by layer
        messages = {}

        for node in G.nodes():
            neighbors = list(G.neighbors(node))

            if not neighbors:
                continue

            # Get neighbor states
            neighbor_states = [node_states[neigh] for neigh in neighbors]

            # Apply different aggregation functions for odd vs even layers
            # This makes it harder to learn immediately but still learnable
            if layer % 2 == 0:
                # Even layers: Mean + std deviation component
                mean_state = np.mean(neighbor_states, axis=0)
                std_state = np.std(neighbor_states, axis=0) + 1e-6  # Avoid division by zero

                # Interaction term using division (harder to approximate)
                messages[node] = mean_state / std_state
            else:
                # Odd layers: Weighted sum based on feature similarity
                node_state = node_states[node]
                weighted_sum = np.zeros_like(node_state)

                # Calculate attention-like weights based on feature similarity
                weights = []
                for neigh, neigh_state in zip(neighbors, neighbor_states):
                    # Non-linear similarity that's harder to approximate
                    sim = np.tanh(np.dot(node_state, neigh_state) / feature_dim)
                    # Penalize dissimilar features exponentially
                    weight = np.exp(sim - 1)
                    weights.append(weight)

                # Normalize weights (but not too aggressively)
                if sum(weights) > 0:
                    weights = [w / (sum(weights) + 0.1 * len(weights)) for w in weights]

                # Apply weights
                for neigh_state, weight in zip(neighbor_states, weights):
                    weighted_sum += neigh_state * weight

                messages[node] = weighted_sum

        # Update phase with layer-specific decay
        layer_decay = decay_factor ** (layer + 1)
        changes = []

        for node in G.nodes():
            if node in messages:
                # Combine previous state with messages using decay
                new_state = (1 - layer_decay) * node_states[node] + layer_decay * messages[node]

                # Add non-linear transformation (harder to learn)
                new_state = np.tanh(new_state) * (1 + np.abs(new_state))

                # Calculate change magnitude (L1 norm)
                change = np.mean(np.abs(new_state - prev_states[node]))
                changes.append(change)

                # Update state
                node_states[node] = new_state

        # Record average change for this layer
        if changes:
            layer_changes.append(np.mean(changes))

    # Final metric calculation from node states
    # Use both magnitude and pattern of message passing results

    # 1. Get final node state statistics
    final_states = np.array(list(node_states.values()))
    state_mean = np.mean(final_states, axis=0)
    state_std = np.std(final_states, axis=0) + 1e-6  # Avoid division by zero

    # 2. Calculate feature coherence (sensitive to edge patterns)
    coherence = np.mean(state_std / (np.abs(state_mean) + 1e-6))

    # 3. Incorporate layer change dynamics (sensitive to message flows)
    # Non-normalized sum (sensitive to missing edges in corrupted graphs)
    dynamics = np.sum(layer_changes) * (num_layers / 3.0)

    # 4. Add structured noise based on node count (makes it harder to learn immediately)
    noise_component = np.sin(len(G.nodes()) / 10) * 0.05

    # 5. Combine components with non-linear interactions
    result = coherence * (1 + np.tanh(dynamics)) + noise_component

    return float(result)


# testing
import networkx as nx
import numpy as np

# Generate a sample graph
G = nx.erdos_renyi_graph(5, 0.5)
for node in G.nodes():
    G.nodes[node]['x'] = np.random.rand(2)  # Random features

r = custom_message_passing_metric(G, num_layers=2, decay_factor=0.7, feature_dim=2)
print(r)
