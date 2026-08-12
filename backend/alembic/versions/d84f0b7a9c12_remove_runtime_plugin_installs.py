"""remove runtime plugin installations

Revision ID: d84f0b7a9c12
Revises: c72e6d5a103b
Create Date: 2026-08-12 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d84f0b7a9c12"
down_revision: str | Sequence[str] | None = "c72e6d5a103b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_table("pipeline_installations")


def downgrade() -> None:
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
