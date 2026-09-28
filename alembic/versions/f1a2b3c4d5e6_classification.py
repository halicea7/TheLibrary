"""Classification: a level on volumes, a portion mark on passages

Revision ID: f1a2b3c4d5e6
Revises: e0f1a2b3c4d5
Create Date: 2026-09-28 12:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, Sequence[str], None] = 'e0f1a2b3c4d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # A volume's level (an id on the scale in classification.json) and where it came from:
    # "manual", "marking" (a banner line in its text), or "cartridge". Null: the default.
    op.add_column('document', sa.Column('classification', sa.String(length=24), nullable=True))
    op.add_column('document', sa.Column('classification_source', sa.String(length=12), nullable=True))
    # A passage's own portion mark ("(S) ..."), where the document has them.
    op.add_column('chunk', sa.Column('portion', sa.String(length=24), nullable=True))
    # The highest level a conversation may draw on (null: no ceiling of its own).
    op.add_column('conversation', sa.Column('ceiling', sa.String(length=24), nullable=True))


def downgrade() -> None:
    op.drop_column('conversation', 'ceiling')
    op.drop_column('chunk', 'portion')
    op.drop_column('document', 'classification_source')
    op.drop_column('document', 'classification')
