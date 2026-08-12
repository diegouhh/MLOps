"""add prediction tracking

Revision ID: 65f2c1a91d9a
Revises: ae68e127a82b
Create Date: 2026-08-11 05:20:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "65f2c1a91d9a"
down_revision: str | Sequence[str] | None = "ae68e127a82b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "prediction_jobs",
        sa.Column("mlflow_run_id", sa.String(length=100), nullable=True),
    )
    op.create_index(
        op.f("ix_prediction_jobs_mlflow_run_id"),
        "prediction_jobs",
        ["mlflow_run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_prediction_jobs_mlflow_run_id"), table_name="prediction_jobs")
    op.drop_column("prediction_jobs", "mlflow_run_id")
