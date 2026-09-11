"""pin concrete model version on prediction jobs

Revision ID: f31a0c9d4e22
Revises: d84f0b7a9c12
Create Date: 2026-09-11 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f31a0c9d4e22"
down_revision: str | Sequence[str] | None = "d84f0b7a9c12"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "prediction_jobs",
        sa.Column("resolved_model_version", sa.String(length=50), nullable=True),
    )
    op.create_index(
        "ix_prediction_jobs_resolved_model_version",
        "prediction_jobs",
        ["resolved_model_version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_prediction_jobs_resolved_model_version",
        table_name="prediction_jobs",
    )
    op.drop_column("prediction_jobs", "resolved_model_version")
