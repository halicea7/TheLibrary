"""Where a volume's subject tag came from

Revision ID: d9e0f1a2b3c4
Revises: c8d9e0f1a2b3
Create Date: 2026-09-27 19:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd9e0f1a2b3c4'
down_revision: Union[str, Sequence[str], None] = 'c8d9e0f1a2b3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # "reader": Tier 1 named it a subject. "shelf": placement added the volume's home as a
    # tag. Null: from before this was kept (library.shelving.classify_tags sorts them out).
    op.add_column('document_category', sa.Column('origin', sa.String(length=8), nullable=True))


def downgrade() -> None:
    op.drop_column('document_category', 'origin')
