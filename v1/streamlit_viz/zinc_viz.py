import streamlit as st
import networkx as nx
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import tempfile
import os
from pyvis.network import Network
import community as community_louvain
import sys
import os.path as osp

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now import from the zinc directory
from zinc.zinc_corruptor import CorruptedZINC


# Load the dataset
@st.cache_resource
def load_dataset(dataset_path):
    """Load the corrupted ZINC dataset from the given path"""
    try:
        dataset = CorruptedZINC(root=dataset_path, subset=True)
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


# Convert PyG graph to NetworkX format
def pyg_to_nx(data, use_original=False):
    """Convert PyG graph to NetworkX format"""
    G = nx.Graph()

    # Add nodes with attributes
    for i in range(data.num_nodes):
        # Store node features
        G.add_node(i, x=data.x[i].numpy())

    # Select proper edges based on whether we want original or corrupted
    if use_original and hasattr(data, 'org_edge_index'):
        edge_index = data.org_edge_index
    else:
        edge_index = data.edge_index

    # Add edges
    for i in range(edge_index.shape[1]):
        # if i % 2 == 0:  # Only add one direction for undirected graph
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        G.add_edge(src, dst)

    return G


# Get consistent positions for both graphs
def get_consistent_positions(G, seed=42):
    # Use community detection to identify clusters
    try:
        partition = community_louvain.best_partition(G)
    except:
        # Fall back to basic spring layout if community detection fails
        return nx.spring_layout(G, seed=seed, k=1.8)

    # Create layout with community awareness
    pos = nx.spring_layout(G, seed=seed, k=1.8)

    return pos


def visualize_graph_pyvis(G, title, positions, height="600px", width="100%"):
    """Visualize graph using PyVis network"""
    # Configure Network
    net = Network(height=height, width=width, notebook=False, bgcolor="#ffffff", font_color="black")

    # Disable physics for stable visualization with fixed positions
    net.toggle_physics(False)

    # Set options
    net.set_options("""
    {
      "nodes": {
        "size": 6,
        "borderWidth": 1,
        "borderWidthSelected": 2,
        "font": {"size": 10, "color": "#333"}
      },
      "edges": {
        "width": 0.8,
        "selectionWidth": 1.5,
        "smooth": false
      },
      "interaction": {
        "hover": true,
        "zoomView": true
      }
    }
    """)

    # Add nodes with consistent positions and coloring by degree
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if len(degrees) > 0 else 1

    for node in G.nodes():
        # Color by degree
        degree = degrees[node]
        # Color from blue (low degree) to red (high degree)
        intensity = int(255 * (degree / max_degree))
        color = f'#{intensity:02x}00{255 - intensity:02x}'

        x, y = positions[node]
        x *= 100  # Scale for PyVis
        y *= 100

        net.add_node(
            node,
            label=f"{node}",
            title=f"Node {node}\nDegree: {degree}",
            color=color,
            x=x,
            y=y,
            physics=False  # Keep position fixed
        )

    # Add edges
    for source, target in G.edges():
        net.add_edge(source, target)

    # Generate HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        net.save_graph(tmpfile.name)
        return tmpfile.name


def main():
    st.set_page_config(layout="wide", page_title="Corrupted ZINC Visualization")
    st.title("Corrupted ZINC Dataset Visualization")

    # Paths to available datasets
    st.sidebar.header("Dataset Selection")
    dataset_path = st.sidebar.text_input(
        "Enter path to corrupted ZINC dataset",
        value="../zinc/corrupted_zinc/zinc_subset_corruption0.50"
    )

    # Load dataset
    dataset = load_dataset(dataset_path)

    if dataset is None:
        st.error(f"Could not load dataset from {dataset_path}. Please check the path and try again.")
        return

    # Graph selection
    st.sidebar.header("Graph Selection")
    selected_idx = st.sidebar.number_input(
        "Select graph ID:",
        min_value=0,
        max_value=len(dataset) - 1,
        value=0
    )

    # Get graph data
    graph_data = dataset[selected_idx]

    # Display graph info
    st.sidebar.header("Graph Information")
    st.sidebar.write(f"Nodes: {graph_data.num_nodes}")
    st.sidebar.write(f"Original Edges: {graph_data.org_edge_index.shape[1] // 2}")
    st.sidebar.write(f"Corrupted Edges: {graph_data.edge_index.shape[1] // 2}")

    # Calculate percentage of edges changed
    if hasattr(graph_data, 'org_edge_index'):
        # Convert to sets for efficient comparison
        orig_edges = set()
        for i in range(graph_data.org_edge_index.shape[1]):
            u = graph_data.org_edge_index[0, i].item()
            v = graph_data.org_edge_index[1, i].item()
            orig_edges.add((min(u, v), max(u, v)))  # Undirected edge

        new_edges = set()
        for i in range(graph_data.edge_index.shape[1]):
            u = graph_data.edge_index[0, i].item()
            v = graph_data.edge_index[1, i].item()
            new_edges.add((min(u, v), max(u, v)))  # Undirected edge

        # Count edges that were added and removed
        removed_edges = orig_edges - new_edges
        added_edges = new_edges - orig_edges

        st.sidebar.write(f"Edges removed: {len(removed_edges)}")
        st.sidebar.write(f"Edges added: {len(added_edges)}")

        # Calculate corruption percentage
        if len(orig_edges) > 0:
            corruption_pct = (len(removed_edges) / len(orig_edges)) * 100
            st.sidebar.write(f"Corruption level: {corruption_pct:.1f}%")

    # Convert to NetworkX
    G_original = pyg_to_nx(graph_data, use_original=True)
    G_corrupted = pyg_to_nx(graph_data, use_original=False)

    # Compute consistent positions
    positions = get_consistent_positions(G_original)

    # Add legend
    st.markdown("""
    <div style="background-color:#f0f0f0; padding:10px; border-radius:5px; margin-bottom:15px">
        <b>Legend:</b> Node colors represent node degree (blue=low degree, red=high degree).<br>
        Hover over nodes to see detailed information.
    </div>
    """, unsafe_allow_html=True)

    # Create side-by-side visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Graph")
        html_file_orig = visualize_graph_pyvis(
            G_original, "Original Graph", positions, height="600px", width="100%"
        )
        st.components.v1.html(open(html_file_orig, 'r').read(), height=600)

    with col2:
        st.subheader("Corrupted Graph")
        html_file_corrupted = visualize_graph_pyvis(
            G_corrupted, "Corrupted Graph", positions, height="600px", width="100%"
        )
        st.components.v1.html(open(html_file_corrupted, 'r').read(), height=600)

    # Add toggle visualization
    st.header("Interactive Toggle Visualization")
    st.write("Toggle between original and corrupted graph to see edge changes")

    show_corrupted = st.toggle("Show Corrupted Graph", value=False)

    # Display either original or corrupted based on toggle state
    if show_corrupted:
        st.subheader("Showing Corrupted Graph")
        html_file_toggle = visualize_graph_pyvis(
            G_corrupted, "Corrupted Graph", positions, height="800px", width="100%"
        )
    else:
        st.subheader("Showing Original Graph")
        html_file_toggle = visualize_graph_pyvis(
            G_original, "Original Graph", positions, height="800px", width="100%"
        )

    st.components.v1.html(open(html_file_toggle, 'r').read(), height=800)

    # Clean up temp files
    os.unlink(html_file_orig)
    os.unlink(html_file_corrupted)
    os.unlink(html_file_toggle)


if __name__ == "__main__":
    main()
