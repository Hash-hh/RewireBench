import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from pyg_dataset import SyntheticRewiringDataset
import seaborn as sns
from scipy import stats


def load_app():
    # Set page to wide mode but not too extreme
    st.set_page_config(layout="wide", page_title="RewiredBench Explorer")

    st.title("RewiredBench Dataset Explorer")

    # Load the dataset
    @st.cache_resource
    def load_data():
        dataset = SyntheticRewiringDataset(root='../rewire_bench')
        return dataset

    with st.spinner('Loading dataset...'):
        dataset = load_data()

    st.success(f"Dataset loaded with {len(dataset)} graphs")

    # Basic dataset statistics
    st.header("Dataset Overview")

    # Get basic statistics
    num_nodes_list = [data.num_nodes for data in dataset]
    num_clusters_list = [data.num_clusters for data in dataset]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Graphs", f"{len(dataset)}")
    with col2:
        st.metric("Avg Nodes per Graph", f"{np.mean(num_nodes_list):.1f}")
    with col3:
        st.metric("Avg Clusters per Graph", f"{np.mean(num_clusters_list):.1f}")

    # Determine number of metrics
    num_metrics = dataset[0].y.shape[0]

    # Metric selection
    metric_names = [f"Metric {i + 1}" for i in range(num_metrics)]
    selected_metric = st.selectbox("Select a metric to analyze:",
                                   range(num_metrics),
                                   format_func=lambda x: metric_names[x])

    # Extract data for the selected metric
    original_vals = torch.stack([d.y[selected_metric] for d in dataset]).numpy()
    rewired_vals = torch.stack([d.y_rewire[selected_metric] for d in dataset]).numpy()

    # Calculate difference between original and rewired
    diff_vals = original_vals - rewired_vals

    # Display metric statistics
    st.subheader(f"Statistics for {metric_names[selected_metric]}")

    stat_cols = st.columns(3)

    with stat_cols[0]:
        st.markdown("**Original Values**")
        st.metric("Mean", f"{np.mean(original_vals):.4f}")
        st.metric("Std Dev", f"{np.std(original_vals):.4f}")
        st.metric("Min", f"{np.min(original_vals):.4f}")
        st.metric("Max", f"{np.max(original_vals):.4f}")

    with stat_cols[1]:
        st.markdown("**Rewired Values**")
        st.metric("Mean", f"{np.mean(rewired_vals):.4f}")
        st.metric("Std Dev", f"{np.std(rewired_vals):.4f}")
        st.metric("Min", f"{np.min(rewired_vals):.4f}")
        st.metric("Max", f"{np.max(rewired_vals):.4f}")

    with stat_cols[2]:
        st.markdown("**Difference (Orig - Rewired)**")
        st.metric("Mean", f"{np.mean(diff_vals):.4f}")
        st.metric("Std Dev", f"{np.std(diff_vals):.4f}")
        st.metric("Min", f"{np.min(diff_vals):.4f}")
        st.metric("Max", f"{np.max(diff_vals):.4f}")

    # Create histograms with better proportions
    st.subheader("Histogram Comparison")

    # Adjusted figure size for better fit
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Original values histogram with more bins
    sns.histplot(original_vals, kde=True, ax=ax1, color='blue', bins=20)
    ax1.set_title("Original Values", fontsize=14)
    ax1.set_xlabel("Value", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.tick_params(labelsize=10)

    # Rewired values histogram
    sns.histplot(rewired_vals, kde=True, ax=ax2, color='green', bins=20)
    ax2.set_title("Rewired Values", fontsize=14)
    ax2.set_xlabel("Value", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.tick_params(labelsize=10)

    # Difference histogram
    sns.histplot(diff_vals, kde=True, ax=ax3, color='red', bins=20)
    ax3.set_title("Difference (Original - Rewired)", fontsize=14)
    ax3.set_xlabel("Value", fontsize=12)
    ax3.set_ylabel("Frequency", fontsize=12)
    ax3.tick_params(labelsize=10)

    plt.tight_layout()
    st.pyplot(fig)

    # Add scatter plot to show correlation
    st.subheader("Original vs Rewired Correlation")

    # Create scatter plot with better proportions
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate correlation
    r_value = np.corrcoef(original_vals, rewired_vals)[0, 1]

    # Create scatter plot with regression line
    sns.regplot(x=original_vals, y=rewired_vals, ax=ax, scatter_kws={'alpha': 0.3, 's': 60})
    ax.set_title(f"Original vs Rewired Values (r = {r_value:.4f})", fontsize=16)
    ax.set_xlabel("Original Values", fontsize=14)
    ax.set_ylabel("Rewired Values", fontsize=14)
    ax.tick_params(labelsize=12)

    # Add diagonal line for reference
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

    plt.tight_layout()
    st.pyplot(fig)

    # Statistical test results
    st.subheader("Statistical Tests")

    test_cols = st.columns(2)

    with test_cols[0]:
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(original_vals, rewired_vals)

        st.write(f"**Paired t-test**: t = {t_stat:.4f}, p-value = {p_value:.6f}")
        if p_value < 0.05:
            st.write("The original and rewired values are significantly different")
        else:
            st.write("No significant difference between original and rewired values")

    with test_cols[1]:
        # Effect size (Cohen's d)
        d = (np.mean(original_vals) - np.mean(rewired_vals)) / np.sqrt(
            (np.var(original_vals) + np.var(rewired_vals)) / 2)
        st.write(f"**Effect size (Cohen's d)**: d = {d:.4f}")

        # Add interpretation of effect size
        if abs(d) < 0.2:
            st.write("Effect size interpretation: Negligible effect")
        elif abs(d) < 0.5:
            st.write("Effect size interpretation: Small effect")
        elif abs(d) < 0.8:
            st.write("Effect size interpretation: Medium effect")
        else:
            st.write("Effect size interpretation: Large effect")

    # Create a sidebar for graph selection
    with st.sidebar:
        st.subheader("Sample Graph Viewer")
        selected_graph = st.slider("Select a graph:", 0, len(dataset) - 1, 0)

        # Show selected graph's metrics
        st.write(f"Graph {selected_graph} metrics:")
        st.write(f"Nodes: {dataset[selected_graph].num_nodes}")
        st.write(f"Clusters: {dataset[selected_graph].num_clusters}")

        # Show full metric values
        st.write("Original metric values:")
        st.write(dataset[selected_graph].y.numpy())

        st.write("Rewired metric values:")
        st.write(dataset[selected_graph].y_rewire.numpy())


if __name__ == "__main__":
    load_app()
