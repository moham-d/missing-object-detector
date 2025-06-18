import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the CSV file
df = pd.read_csv("mod_comparison.csv")

# Define bins
bins = np.linspace(0, 1.0, 21)

# Function to create and show histogram with custom styling


def plot_hist_melted_custom_style(data_normal, data_defective, x_label, color_palette, filename):
    # Prepare DataFrame
    df_plot = pd.DataFrame({
        'Normal': data_normal,
        'Defective': data_defective
    })
    df_melted = df_plot.melt(var_name="Label", value_name="Similarity")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(
        data=df_melted,
        x="Similarity",
        hue="Label",
        bins=bins,
        element="step",
        stat="count",
        common_norm=False,
        palette=color_palette,
        ax=ax
    )

    # Clean axis styling
    ax.set_title("")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of Images")

    # Remove spines except bottom
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 1.0)

    # Preserve ticks
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis='y', left=True)
    ax.xaxis.set_ticks_position("bottom")
    ax.grid(False)

    # Manual legend (no frame, no title)
    handles = [
        mpatches.Patch(color=color_palette[0], label="Normal"),
        mpatches.Patch(color=color_palette[1], label="Defective")
    ]
    legend = ax.legend(handles=handles, title=None)
    legend.set_frame_on(False)

    sns.despine()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# YOLO ANNOTATED
plot_hist_melted_custom_style(
    data_normal=df[df['normal'] == True]['node_jaccard'],
    data_defective=df[df['normal'] == False]['node_jaccard'],
    x_label="Node Jaccard Similarity",
    color_palette=["#D81B60", "#1E88E5"],
    filename="figures/distribution/yolo_node_jaccard.png"
)

plot_hist_melted_custom_style(
    data_normal=df[df['normal'] == True]['edge_jaccard'],
    data_defective=df[df['normal'] == False]['edge_jaccard'],
    x_label="Edge Jaccard Similarity",
    color_palette=["#D81B60", "#1E88E5"],
    filename="figures/distribution/yolo_edge_jaccard.png"
)

# CVAT ANNOTATED
plot_hist_melted_custom_style(
    data_normal=df[df['normal'] == True]['cvat_node_jaccard'],
    data_defective=df[df['normal'] == False]['cvat_node_jaccard'],
    x_label="Node Jaccard Similarity",
    color_palette=["#FFC107", "#004D40"],
    filename="figures/distribution/cvat_node_jaccard.png"
)

plot_hist_melted_custom_style(
    data_normal=df[df['normal'] == True]['cvat_edge_jaccard'],
    data_defective=df[df['normal'] == False]['cvat_edge_jaccard'],
    x_label="Edge Jaccard Similarity",
    color_palette=["#FFC107", "#004D40"],
    filename="figures/distribution/cvat_edge_jaccard.png"
)
