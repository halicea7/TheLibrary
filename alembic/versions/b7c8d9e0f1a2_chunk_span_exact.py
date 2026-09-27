"""A chunk says whether its span is exact

Revision ID: b7c8d9e0f1a2
Revises: a5b6c7d8e9f0
Create Date: 2026-09-27 18:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b7c8d9e0f1a2'
down_revision: Union[str, Sequence[str], None] = 'a5b6c7d8e9f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # True: text[char_start:char_end] of the extraction holds this chunk (whitespace aside).
    # False: the location is approximate. Null: not yet checked (chunks from before this).
    op.add_column('chunk', sa.Column('span_exact', sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column('chunk', 'span_exact')
