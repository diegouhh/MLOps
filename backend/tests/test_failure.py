from __future__ import annotations

import pytest

from app.schemas.api import ExperimentCreate
from app.services.experiments import create_experiment, execute_experiment, get_experiment


def test_failed_stage_is_persisted_without_blocking_platform(db, settings, demo_dataset):
    experiment = create_experiment(
        db,
        ExperimentCreate(
            name="Controlled failure",
            dataset_id=demo_dataset.id,
            pipeline_id="tabular_basic",
            pipeline_config={"target_column": "does_not_exist"},
            models=[{"model_id": "logistic_regression", "parameters": {}}],
        ),
    )
    with pytest.raises(Exception, match="no existe"):
        execute_experiment(db, settings, experiment.id)
    persisted = get_experiment(db, experiment.id)
    assert persisted.status == "failed"
    assert persisted.runs[0].status == "failed"
    assert persisted.error_message
