"""pin dataset version and registry model name

Revision ID: b61f4a9d8e20
Revises: 65f2c1a91d9a
Create Date: 2026-08-11 10:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b61f4a9d8e20"
down_revision: str | Sequence[str] | None = "65f2c1a91d9a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("experiments") as batch_op:
        batch_op.add_column(
            sa.Column("dataset_version", sa.Integer(), nullable=False, server_default="1")
        )
        batch_op.add_column(
            sa.Column("registered_model_name", sa.String(length=200), nullable=True)
        )
        batch_op.create_index(
            "ix_experiments_registered_model_name",
            ["registered_model_name"],
            unique=False,
        )

    connection = op.get_bind()
    connection.execute(
        sa.text(
            "UPDATE experiments "
            "SET registered_model_name = 'neuroops-' || pipeline_id "
            "WHERE registered_model_name IS NULL"
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("experiments") as batch_op:
        batch_op.drop_index("ix_experiments_registered_model_name")
        batch_op.drop_column("registered_model_name")
        batch_op.drop_column("dataset_version")
