import streamlit as st
import torch
import numpy as np
from v1.pyg_dataset import SyntheticRewiringDataset
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_app():
    # Set page to wide mode but not too extreme
    st.set_page_config(layout="wide", page_title="RewiredBench Explorer")

    st.title("RewiredBench Dataset Explorer")

    # Load the dataset
    @st.cache_resource
    def load_data():
        dataset = SyntheticRewiringDataset(root='../rewire_bench_mp1')
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

    # Create interactive subplots with Plotly
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Original Values", "Rewired Values", "Difference (Original - Rewired)"],
                        horizontal_spacing=0.1)

    # Add checkbox to toggle log scale
    use_log_scale = st.checkbox("Use logarithmic scale for histograms", value=False)

    # Add slider to limit the range
    q_min, q_max = st.slider(
        "Limit histogram range (percentile):",
        min_value=0.0,
        max_value=100.0,
        value=[0.0, 100.0],
        step=1.0
    )

    # Add a slider for bin count
    bin_count = st.slider("Number of bins:", min_value=10, max_value=500, value=30, step=5)

    # Calculate range limits based on percentiles
    if q_min > 0 or q_max < 100:
        orig_min = np.percentile(original_vals, q_min)
        orig_max = np.percentile(original_vals, q_max)
        rew_min = np.percentile(rewired_vals, q_min)
        rew_max = np.percentile(rewired_vals, q_max)
        diff_min = np.percentile(diff_vals, q_min)
        diff_max = np.percentile(diff_vals, q_max)

        # Filter values to only include those within the range
        orig_filtered = original_vals[(original_vals >= orig_min) & (original_vals <= orig_max)]
        rew_filtered = rewired_vals[(rewired_vals >= rew_min) & (rewired_vals <= rew_max)]
        diff_filtered = diff_vals[(diff_vals >= diff_min) & (diff_vals <= diff_max)]
    else:
        orig_min = orig_max = rew_min = rew_max = diff_min = diff_max = None
        orig_filtered = original_vals
        rew_filtered = rewired_vals
        diff_filtered = diff_vals

    # Original values histogram
    fig.add_trace(
        go.Histogram(
            x=original_vals,
            nbinsx=bin_count,
            name="Original",
            marker_color='blue',
            opacity=0.7,
            xbins=dict(
                start=orig_min,
                end=orig_max
            )
        ),
        row=1, col=1
    )

    # Use the filtered data for calculating the mean line height
    if len(orig_filtered) > 0:
        hist_counts, _ = np.histogram(orig_filtered, bins=bin_count)
        max_height = max(hist_counts) if len(hist_counts) > 0 else 1
        fig.add_trace(
            go.Scatter(x=[np.mean(orig_filtered), np.mean(orig_filtered)],
                       y=[0, max_height],
                       mode="lines", name="Mean (Original)", line=dict(color="darkblue", width=2, dash="dash")),
            row=1, col=1
        )

    # Rewired values histogram
    fig.add_trace(
        go.Histogram(
            x=rewired_vals,
            nbinsx=bin_count,
            name="Rewired",
            marker_color='green',
            opacity=0.7,
            xbins=dict(
                start=rew_min,
                end=rew_max
            )
        ),
        row=1, col=2
    )

    # Use the filtered data for calculating the mean line height
    if len(rew_filtered) > 0:
        hist_counts, _ = np.histogram(rew_filtered, bins=bin_count)
        max_height = max(hist_counts) if len(hist_counts) > 0 else 1
        fig.add_trace(
            go.Scatter(x=[np.mean(rew_filtered), np.mean(rew_filtered)],
                       y=[0, max_height],
                       mode="lines", name="Mean (Rewired)", line=dict(color="darkgreen", width=2, dash="dash")),
            row=1, col=2
        )

    # Difference histogram
    fig.add_trace(
        go.Histogram(
            x=diff_vals,
            nbinsx=bin_count,
            name="Difference",
            marker_color='red',
            opacity=0.7,
            xbins=dict(
                start=diff_min,
                end=diff_max
            )
        ),
        row=1, col=3
    )

    # Use the filtered data for calculating the mean line height
    if len(diff_filtered) > 0:
        hist_counts, _ = np.histogram(diff_filtered, bins=bin_count)
        max_height = max(hist_counts) if len(hist_counts) > 0 else 1
        fig.add_trace(
            go.Scatter(x=[np.mean(diff_filtered), np.mean(diff_filtered)],
                       y=[0, max_height],
                       mode="lines", name="Mean (Difference)", line=dict(color="darkred", width=2, dash="dash")),
            row=1, col=3
        )
    fig.add_trace(
        go.Scatter(x=[np.mean(diff_vals), np.mean(diff_vals)],
                   y=[0, np.histogram(diff_vals, bins=20)[0].max()],
                   mode="lines", name="Mean (Difference)", line=dict(color="darkred", width=2, dash="dash")),
        row=1, col=3
    )

    # Update layout for better appearance
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="",
        xaxis_title_text="Value",
        yaxis_title_text="Frequency",
    )

    # Update x and y axis labels for each subplot
    for i in range(1, 4):
        fig.update_xaxes(title_text="Value", row=1, col=i)
        fig.update_yaxes(title_text="Frequency" if i == 1 else "", row=1, col=i)

        # Apply log scale if selected
        if use_log_scale:
            fig.update_yaxes(type="log", row=1, col=i)

    # Display the interactive plot
    st.plotly_chart(fig, use_container_width=True)


    # Add scatter plot to show correlation
    st.subheader("Original vs Rewired Correlation")

    # Calculate correlation
    r_value = np.corrcoef(original_vals, rewired_vals)[0, 1]

    # Create interactive scatter plot
    scatter_fig = px.scatter(
        x=original_vals,
        y=rewired_vals,
        opacity=0.6,
        labels={"x": "Original Values", "y": "Rewired Values"},
        title=f"Original vs Rewired Values (r = {r_value:.4f})",
        trendline="ols",  # Add ordinary least squares regression line
        height=600
    )

    # Add diagonal reference line (y=x)
    x_range = [min(min(original_vals), min(rewired_vals)), max(max(original_vals), max(rewired_vals))]
    scatter_fig.add_trace(
        go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            name="y=x",
            line=dict(color="black", dash="dash", width=1),
            opacity=0.5
        )
    )

    # Use same percentile filtering as histograms
    if q_min > 0 or q_max < 100:
        # Add a button to toggle between full range and filtered range
        show_full_range = st.checkbox("Show full range in scatter plot", value=False)

        if not show_full_range:
            # Filter scatter plot to use same range as histograms
            orig_min = np.percentile(original_vals, q_min)
            orig_max = np.percentile(original_vals, q_max)
            rew_min = np.percentile(rewired_vals, q_min)
            rew_max = np.percentile(rewired_vals, q_max)

            scatter_fig.update_layout(
                xaxis=dict(range=[orig_min, orig_max]),
                yaxis=dict(range=[rew_min, rew_max])
            )

    # Improve plot appearance
    scatter_fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="darkgray",
            zerolinewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="darkgray",
            zerolinewidth=1
        )
    )

    # Display the scatter plot
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Add hover info to show points that are outliers
    scatter_fig.update_traces(
        hovertemplate="<b>Original:</b> %{x:.4f}<br><b>Rewired:</b> %{y:.4f}<br><b>Difference:</b> %{customdata:.4f}",
        customdata=original_vals - rewired_vals
    )

    # Add annotations to show statistical details
    scatter_fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Correlation: r = {r_value:.4f}",
        showarrow=False,
        font=dict(size=14),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4
    )

    # # Display the interactive plot
    # st.plotly_chart(scatter_fig, use_container_width=True)

    # # Statistical test results
    # st.subheader("Statistical Tests")
    #
    # test_cols = st.columns(2)
    #
    # with test_cols[0]:
    #     # Paired t-test
    #     t_stat, p_value = stats.ttest_rel(original_vals, rewired_vals)
    #
    #     st.write(f"**Paired t-test**: t = {t_stat:.4f}, p-value = {p_value:.6f}")
    #     if p_value < 0.05:
    #         st.write("The original and rewired values are significantly different")
    #     else:
    #         st.write("No significant difference between original and rewired values")
    #
    # with test_cols[1]:
    #     # Effect size (Cohen's d)
    #     d = (np.mean(original_vals) - np.mean(rewired_vals)) / np.sqrt(
    #         (np.var(original_vals) + np.var(rewired_vals)) / 2)
    #     st.write(f"**Effect size (Cohen's d)**: d = {d:.4f}")
    #
    #     # Add interpretation of effect size
    #     if abs(d) < 0.2:
    #         st.write("Effect size interpretation: Negligible effect")
    #     elif abs(d) < 0.5:
    #         st.write("Effect size interpretation: Small effect")
    #     elif abs(d) < 0.8:
    #         st.write("Effect size interpretation: Medium effect")
    #     else:
    #         st.write("Effect size interpretation: Large effect")
    #
    # # Create a sidebar for graph selection
    # with st.sidebar:
    #     st.subheader("Sample Graph Viewer")
    #     selected_graph = st.slider("Select a graph:", 0, len(dataset) - 1, 0)
    #
    #     # Show selected graph's metrics
    #     st.write(f"Graph {selected_graph} metrics:")
    #     st.write(f"Nodes: {dataset[selected_graph].num_nodes}")
    #     st.write(f"Clusters: {dataset[selected_graph].num_clusters}")
    #
    #     # Show full metric values
    #     st.write("Original metric values:")
    #     st.write(dataset[selected_graph].y.numpy())
    #
    #     st.write("Rewired metric values:")
    #     st.write(dataset[selected_graph].y_rewire.numpy())


if __name__ == "__main__":
    load_app()
