"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


# def register_pipelines() -> dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines

from mod_project.pipelines.dataset_pipeline import pipeline as dataset_pipeline
from mod_project.pipelines.mod_pipeline import pipeline as mod_pipeline
from mod_project.pipelines.mod_evaluation_pipeline import pipeline as eval_pipeline


def register_pipelines():
    return {
        "dataset": dataset_pipeline.create_pipeline(),
        "mod": mod_pipeline.create_pipeline(),
        "eval": eval_pipeline.create_pipeline(),
        "__default__": dataset_pipeline.create_pipeline()
    }
