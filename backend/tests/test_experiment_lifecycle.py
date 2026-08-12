from __future__ import annotations

from app.schemas.api import ExperimentCreate
from app.services.experiments import cancel_experiment, create_experiment, rerun_experiment


def _payload(dataset_id: str) -> ExperimentCreate:
    return ExperimentCreate(
        name="Lifecycle",
        dataset_id=dataset_id,
        pipeline_id="tabular_basic",
        pipeline_config={"target_column": "species"},
        models=[{"model_id": "logistic_regression", "parameters": {}}],
        random_seed=19,
    )


def test_queued_experiment_can_be_cancelled(db, demo_dataset):
    experiment = create_experiment(db, _payload(demo_dataset.id))
    cancelled = cancel_experiment(db, experiment.id)
    assert cancelled.status == "cancelled"
    assert cancelled.runs[0].status == "cancelled"


def test_rerun_preserves_reproducible_configuration(db, demo_dataset):
    source = create_experiment(db, _payload(demo_dataset.id))
    rerun = rerun_experiment(db, source.id)
    assert rerun.id != source.id
    assert rerun.dataset_id == source.dataset_id
    assert rerun.pipeline_config == source.pipeline_config
    assert rerun.model_parameters == source.model_parameters
    assert rerun.random_seed == 19
