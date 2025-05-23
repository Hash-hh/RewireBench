import streamlit as st
import networkx as nx
import pandas as pd
from v1.pyg_dataset import SyntheticRewiringDataset
import tempfile
import os
from pyvis.network import Network
import community as community_louvain


# Load the dataset
@st.cache_resource
def load_dataset():
    return SyntheticRewiringDataset(root='../rewire_bench_easy')


# Convert PyG graph to NetworkX format
def pyg_to_nx(data, use_original=False):
    G = nx.Graph()

    # Add nodes with attributes
    for i in range(data.num_nodes):
        # Store the whole feature vector
        node_features = data.x[i].numpy()
        # Add block/cluster information from node_cluster_labels
        block = int(data.node_cluster_labels[i].item()) if hasattr(data, 'node_cluster_labels') else None
        G.add_node(i, x=node_features, block=block)

    # Select proper edges based on whether we want original or rewired
    if use_original:
        edge_index = data.org_edge_index
        edge_attr = data.org_edge_attr
    else:
        edge_index = data.edge_index
        edge_attr = data.edge_attr

    # Add edges with attributes
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        # Store edge attributes
        attr = {
            "is_intra": edge_attr[i, 0].item() == 1,
            "edge_attr": edge_attr[i].numpy()
        }
        G.add_edge(src, dst, **attr)

    return G


# Get consistent positions for both graphs with cluster awareness
def get_consistent_positions(G, for_toggle=False):
    # Use community detection to identify clusters
    try:
        partition = community_louvain.best_partition(G)
    except:
        # Fall back to basic spring layout if community detection fails
        return nx.spring_layout(G, seed=42, k=2.0 if for_toggle else 1.8)

    # Create a new graph with additional nodes representing cluster centers
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())

    # Find unique clusters
    clusters = set(partition.values())

    # Add a central node for each cluster
    center_nodes = {}
    for cluster_id in clusters:
        center_node = f"center_{cluster_id}"
        center_nodes[cluster_id] = center_node
        H.add_node(center_node)

        # Connect all nodes in cluster to their center node
        for node, node_cluster in partition.items():
            if node_cluster == cluster_id:
                # Use even weaker connections for toggle view to spread nodes more
                weight = 0.3 if for_toggle else 0.5
                H.add_edge(node, center_node, weight=weight)

    # Get layout with cluster centers attracting their nodes
    # Use higher k and more iterations for toggle view
    k_value = 3.5 if for_toggle else 2.5
    iterations = 150 if for_toggle else 100
    layout = nx.spring_layout(H, seed=42, k=k_value, iterations=iterations)

    # Remove the center nodes from the layout
    final_layout = {node: pos for node, pos in layout.items() if isinstance(node, int)}

    # Scale the positions for better spacing - more for toggle view
    scale = 3.0 if for_toggle else 2.0
    final_layout = {node: pos * scale for node, pos in final_layout.items()}

    return final_layout


def get_node_color_map(G, color_by, feature_idx=0):
    node_color_map = {}

    if color_by == "cluster":
        # Color by cluster/block
        blocks = [G.nodes[i].get('block') for i in range(len(G.nodes))]
        unique_blocks = sorted(set(blocks))

        # Use a predefined distinct color palette
        distinct_colors = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
            '#ff7f00', '#ffff33', '#a65628', '#f781bf',
            '#999999', '#66c2a5', '#fc8d62', '#8da0cb',
            '#e78ac3', '#a6d854', '#ffd92f', '#e5c494'
        ]

        # If we have more blocks than colors, cycle through the colors
        block_to_color = {}
        for i, block in enumerate(unique_blocks):
            block_to_color[block] = distinct_colors[i % len(distinct_colors)]

        # Map nodes to colors
        for i in range(len(G.nodes)):
            block = G.nodes[i].get('block')
            node_color_map[i] = block_to_color.get(block, '#888888')

    else:  # Color by feature
        # Extract selected feature for all nodes
        node_features = [G.nodes[i]['x'][feature_idx] for i in range(len(G.nodes))]
        min_feat, max_feat = min(node_features), max(node_features)

        for i, feat in enumerate(node_features):
            if max_feat > min_feat:
                norm_feat = (feat - min_feat) / (max_feat - min_feat)
            else:
                norm_feat = 0.5

            r = int(255 * norm_feat)
            b = int(255 * (1 - norm_feat))
            node_color_map[i] = f'#{r:02x}00{b:02x}'

    return node_color_map


# Visualize graph with PyVis
def visualize_graph_pyvis(G, title, node_color_map, positions, height="600px", width="100%", is_toggle_view=False):
    # Configure Network
    net = Network(height=height, width=width, notebook=False, bgcolor="#ffffff", font_color="black")

    # Disable physics for stable visualization with fixed positions
    net.toggle_physics(False)

    # Set options - smaller nodes for regular view, slightly larger for toggle view
    node_size = 8 if is_toggle_view else 6
    font_size = 12 if is_toggle_view else 10
    edge_width = 1.0 if is_toggle_view else 0.8

    net.set_options(f"""
    {{
      "nodes": {{
        "size": {node_size},
        "borderWidth": 1,
        "borderWidthSelected": 2,
        "font": {{"size": {font_size}, "color": "#333"}}
      }},
      "edges": {{
        "width": {edge_width},
        "selectionWidth": 1.5,
        "smooth": false
      }},
      "interaction": {{
        "hover": true,
        "zoomView": true
      }}
    }}
    """)

    # Add nodes with consistent positions
    for node, data in G.nodes(data=True):
        color = node_color_map[node]
        x, y = positions[node]
        x *= 100  # Scale for PyVis
        y *= 100

        # Create detailed tooltip with all node features
        features_str = ", ".join([f"f{i}: {val:.2f}" for i, val in enumerate(data['x'])])
        cluster_str = f"Cluster: {data.get('block')}" if data.get('block') is not None else ""

        net.add_node(
            node,
            label=f"{node}",
            title=f"Node {node}\n{cluster_str}\n{features_str}",
            color=color,
            x=x,
            y=y,
            physics=False  # Keep position fixed
        )

    # Color edges based on intra/inter cluster
    for source, target, data in G.edges(data=True):
        color = '#00FF00' if data.get('is_intra', False) else '#FF0000'

        # Create detailed edge tooltip
        edge_type = "Intra-cluster edge" if data.get('is_intra', False) else "Inter-cluster edge"
        edge_attr_str = f"Edge features: {data.get('edge_attr', [])}"

        net.add_edge(source, target, color=color, title=f"{edge_type}\n{edge_attr_str}", width=edge_width)

    # Generate HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        net.save_graph(tmpfile.name)
        return tmpfile.name



def display_metric_names(metric_index):
    metric_names = {
        0: "local_easy1", 1: "local_easy2", 2: "local_easy3",
        3: "local_hard1", 4: "local_hard2", 5: "local_hard3"
    }
    return metric_names.get(metric_index, f"Metric {metric_index + 1}")


def main():
    st.set_page_config(layout="wide", page_title="RewireBench Visualization")
    st.title("RewireBench")

    dataset = load_dataset()

    # Graph selection
    st.sidebar.header("Graph Selection")
    select_method = st.sidebar.radio("Select graph by:", ["Name", "ID"])

    if select_method == "Name":
        graph_names = [f"graph_{i}" for i in range(len(dataset))]
        selected_name = st.sidebar.selectbox("Select graph name:", graph_names)
        selected_idx = int(selected_name.split('_')[1])
    else:
        selected_idx = st.sidebar.number_input("Enter graph ID:", min_value=0, max_value=len(dataset) - 1, value=0)

    # Get graph data
    graph_data = dataset[selected_idx]

    # Display graph info
    st.sidebar.header("Graph Information")
    st.sidebar.write(f"Name: graph_{selected_idx}")
    st.sidebar.write(f"Nodes: {graph_data.num_nodes}")
    st.sidebar.write(f"Clusters: {graph_data.num_clusters}")

    # Display metrics
    if hasattr(graph_data, 'y') and graph_data.y is not None:
        st.sidebar.header("Metrics")
        metrics_df = pd.DataFrame({
            "Metric": [display_metric_names(i) for i in range(len(graph_data.y))],
            "Original": graph_data.y.numpy(),
            "Rewired": graph_data.y_rewire.numpy() if hasattr(graph_data, 'y_rewire') else ["-"] * len(graph_data.y)
        })
        st.sidebar.dataframe(metrics_df)

    # Convert to NetworkX
    G_original = pyg_to_nx(graph_data, use_original=True)
    G_rewired = pyg_to_nx(graph_data, use_original=False)

    # Determine how many features are available
    num_features = len(G_original.nodes[0]['x'])

    # Display options for node coloring
    st.sidebar.header("Visualization Options")
    color_options = ["Cluster"] + [f"Feature {i}" for i in range(num_features)]
    color_by = st.sidebar.selectbox("Color nodes by:", color_options)

    # Convert choice to parameters for coloring function
    if color_by == "Cluster":
        color_mode = "cluster"
        feature_idx = 0
    else:
        color_mode = "feature"
        feature_idx = int(color_by.split(" ")[1])  # Extract feature index from "Feature X"

    # Create node color mapping based on selected feature or cluster
    node_color_map = get_node_color_map(G_original, color_mode, feature_idx)

    # Store positions in session state to maintain them across rerenders
    if 'positions' not in st.session_state or 'last_graph_id' not in st.session_state or st.session_state.last_graph_id != selected_idx:
        st.session_state.positions = get_consistent_positions(G_original)
        st.session_state.last_graph_id = selected_idx

    # Store toggle positions separately for a more relaxed view
    if 'toggle_positions' not in st.session_state or 'last_toggle_graph_id' not in st.session_state or st.session_state.last_toggle_graph_id != selected_idx:
        st.session_state.toggle_positions = get_consistent_positions(G_original, for_toggle=True)
        st.session_state.last_toggle_graph_id = selected_idx

    positions = st.session_state.positions
    toggle_positions = st.session_state.toggle_positions

    # Add legend at the top
    st.header("Graph Visualizations")

    # Create color legend based on selected coloring method
    if color_mode == "feature":
        feature_name = color_by.split(" ")[1]
        legend_html = f"""
        <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom:15px">
            <b>Legend:</b> Node colors show {color_by} values (blue=low, red=high).
            Edge colors: Green=Intra-cluster, Red=Inter-cluster.<br>
            Hover over nodes and edges to see detailed information including all features and cluster membership.
        </div>
        """
    else:  # cluster mode
        legend_html = """
        <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom:15px">
            <b>Legend:</b> Node colors represent different clusters.
            Edge colors: Green=Intra-cluster, Red=Inter-cluster.<br>
            Hover over nodes and edges to see detailed information including all features.
        </div>
        """

    st.markdown(legend_html, unsafe_allow_html=True)

    # Create side-by-side visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Honest Graph")
        html_file_orig = visualize_graph_pyvis(G_original, "Honest Graph", node_color_map, positions, height="600px",
                                               width="100%")
        st.components.v1.html(open(html_file_orig, 'r').read(), height=600)

    with col2:
        st.subheader("Corrupt Graph")
        html_file_rewired = visualize_graph_pyvis(G_rewired, "Corrupt Graph", node_color_map, positions, height="600px",
                                                  width="100%")
        st.components.v1.html(open(html_file_rewired, 'r').read(), height=600)

    # Add toggle visualization below with much larger canvas
    st.header("Interactive Toggle Visualization")
    st.write("Toggle between honest and Corrupt graph to see edge changes")

    show_rewired = st.toggle("Show Corrupt Graph", value=False)

    # Display either Honest or rewired based on toggle state
    # Use toggle_positions for more spacing and a clearer view
    if show_rewired:
        st.subheader("Showing Corrupt Graph")
        html_file_toggle = visualize_graph_pyvis(G_rewired, "Corrupt Graph", node_color_map, toggle_positions,
                                                 height="900px", width="100%", is_toggle_view=True)
    else:
        st.subheader("Showing Honest Graph")
        html_file_toggle = visualize_graph_pyvis(G_original, "Honest Graph", node_color_map, toggle_positions,
                                                 height="900px", width="100%", is_toggle_view=True)

    st.components.v1.html(open(html_file_toggle, 'r').read(), height=900)

    # Clean up temp files
    os.unlink(html_file_orig)
    os.unlink(html_file_rewired)
    os.unlink(html_file_toggle)

if __name__ == "__main__":
    main()
