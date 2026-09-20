"""chunk.kind: text, or a figure read by the vision model

Revision ID: c1f2a3b4d5e6
Revises: 5bbfdac4a146
Create Date: 2026-09-20 01:40:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c1f2a3b4d5e6'
down_revision: Union[str, Sequence[str], None] = '5bbfdac4a146'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('chunk', sa.Column('kind', sa.String(length=16), nullable=False, server_default='text'))


def downgrade() -> None:
    op.drop_column('chunk', 'kind')
