from kedro.pipeline import Pipeline, node
from .nodes import (
    precision_recall_curve,
    pos_threshold_sweep,
    edge_threshold_sweep
)


def create_pipeline(**kwargs):
    return Pipeline([
        node(precision_recall_curve,
             inputs=["mod_combined_dataset", "reference_graphs"],
             outputs=None,
             name="precision_recall_curve"),
        node(pos_threshold_sweep,
             inputs=["mod_combined_dataset", "reference_graphs"],
             outputs=None,
             name="pos_threshold_sweep"),
        node(edge_threshold_sweep,
             inputs=["mod_combined_dataset", "reference_graphs"],
             outputs=None,
             name="edge_threshold_sweep")
    ])
