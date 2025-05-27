from kedro.pipeline import Pipeline, node
from .nodes import (
    load_annotated_data,
    preprocess_for_lama,
    simulate_missing_parts,
    select_normal_images
)


def create_pipeline(**kwargs):
    return Pipeline([
        node(load_annotated_data,
             inputs=["raw_dataset_images", "raw_dataset_labels",
                     "params:class_names"],
             outputs="paired_dataset",
             name="load_annotated_data"
             ),
        node(preprocess_for_lama,
             inputs=["paired_dataset", "params:num_images"],
             outputs=["lama_input_images", "inpainted_labels"],
             name="preprocess_for_lama"
             ),
        node(simulate_missing_parts,
             inputs="lama_input_images",
             outputs="inpainted_dataset",
             name="simulate_missing_parts"),
        node(select_normal_images,
             inputs=["paired_dataset", "inpainted_labels", "params:num_images"],
             outputs=["normal_dataset", "normal_labels"],
             name="select_normal_images"),
    ])
