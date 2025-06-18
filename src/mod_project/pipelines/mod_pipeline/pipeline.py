import time
from functools import wraps

from kedro.pipeline import Pipeline, node
from .nodes import (
    reference_load_data,
    reference_to_graphs,

    mod_load_data,
    mod_detect_objects,
    mod_yolo_to_graphs,
    mod_cvat_to_graphs,
    mod_comparison,
)


def timed_node(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[{func.__name__}] Execution time: {end - start:.2f} seconds")
        return result
    return wrapper


def create_pipeline(**kwargs):
    return Pipeline([
        node(timed_node(reference_load_data),
             inputs=["reference_dataset", "reference_labels"],
             outputs="reference_combined",
             name="reference_load_data"),

        node(timed_node(reference_to_graphs),
             inputs=["reference_combined",
                     "params:distance_treshold", "params:class_names"],
             outputs="reference_graphs",
             name="reference_to_graphs"),

        node(timed_node(mod_load_data),
             inputs=["classified_normal_dataset", "classified_normal_labels",
                     "classified_inpainted_dataset", "classified_inpainted_labels"],
             outputs="mod_combined_dataset",
             name="mod_load_data"),

        node(timed_node(mod_detect_objects),
             inputs="mod_combined_dataset",
             outputs="predicted_bounding_boxes",
             name="mod_detect_objects"),

        node(timed_node(mod_yolo_to_graphs),
             inputs=["mod_combined_dataset", "predicted_bounding_boxes",
                     "params:distance_treshold", "params:class_names"],
             outputs="input_graphs",
             name="mod_to_graphs"),

        node(timed_node(mod_cvat_to_graphs),
             inputs=["mod_combined_dataset", "params:distance_treshold",
                     "params:class_names"],
             outputs="cvat_graphs",
             name="mod_cvat_to_graphs"),

        node(timed_node(mod_comparison),
             inputs=["input_graphs", "cvat_graphs", "reference_graphs",
                     "params:pos_threshold", "params:edge_threshold"],
             outputs=None,
             name="mod_comparison"),
    ])
