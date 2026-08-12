"""persist Prefect-owned background jobs

Revision ID: c72e6d5a103b
Revises: b61f4a9d8e20
Create Date: 2026-08-11 21:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c72e6d5a103b"
down_revision: str | Sequence[str] | None = "b61f4a9d8e20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("prediction_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("prefect_flow_run_id", sa.String(length=100), nullable=True)
        )
        batch_op.create_index(
            "ix_prediction_jobs_prefect_flow_run_id",
            ["prefect_flow_run_id"],
            unique=False,
        )

    op.create_table(
        "pipeline_installations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("pipeline_id", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("prefect_flow_run_id", sa.String(length=100), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_pipeline_installations_pipeline_id",
        "pipeline_installations",
        ["pipeline_id"],
        unique=True,
    )
    op.create_index(
        "ix_pipeline_installations_status",
        "pipeline_installations",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_pipeline_installations_prefect_flow_run_id",
        "pipeline_installations",
        ["prefect_flow_run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_pipeline_installations_prefect_flow_run_id",
        table_name="pipeline_installations",
    )
    op.drop_index("ix_pipeline_installations_status", table_name="pipeline_installations")
    op.drop_index("ix_pipeline_installations_pipeline_id", table_name="pipeline_installations")
    op.drop_table("pipeline_installations")
    with op.batch_alter_table("prediction_jobs") as batch_op:
        batch_op.drop_index("ix_prediction_jobs_prefect_flow_run_id")
        batch_op.drop_column("prefect_flow_run_id")
