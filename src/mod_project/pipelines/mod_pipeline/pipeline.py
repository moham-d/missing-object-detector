from kedro.pipeline import Pipeline, node
from .nodes import (
    load_mod_data,
    detect_objects,
    image_to_graph,
    reference_patterns
)


def create_pipeline(**kwargs):
    return Pipeline([
        node(load_mod_data,
             inputs=["normal_dataset", "normal_labels",
                     "inpainted_dataset", "inpainted_labels"],
             outputs="mod_combined_dataset",
             name="load_mod_data"),
        node(detect_objects,
             inputs="mod_combined_dataset",
             outputs="predicted_bounding_boxes",
             name="detect_objects"),
        #    node(image_to_graph,
        #         inputs=["mod_combined_dataset", "predicted_bounding_boxes",
        #                 "params:distance_treshold", "params:class_names"],
        #         outputs="graph_representation",
        #         name="image_to_graph")
        node(reference_patterns,
             inputs="predicted_bounding_boxes",
             outputs="patterns",
             name="reference_patterns")
    ])
