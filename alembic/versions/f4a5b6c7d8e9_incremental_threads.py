"""Incremental threads: clusters remember their members, claim vectors are kept

Revision ID: f4a5b6c7d8e9
Revises: e3f4a5b6c7d8
Create Date: 2026-09-21 09:30:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import HALFVEC


revision: str = 'f4a5b6c7d8e9'
down_revision: Union[str, Sequence[str], None] = 'e3f4a5b6c7d8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # A fingerprint of exactly which claims a cluster holds: the next rebuild keeps a
    # cluster whose fingerprint it sees again, summary, verdict and all.
    op.add_column('cluster', sa.Column('member_key', sa.String(length=32), nullable=True))
    op.add_column('cluster', sa.Column('judged_key', sa.String(length=64), nullable=True))
    op.create_index('ix_cluster_member_key', 'cluster', ['member_key'])
    # Claim vectors, by the claim's text: sixty thousand of them were re-embedded on every
    # rebuild. Kept apart from the retrieval index, which they have no business in.
    op.create_table(
        'claim_vector',
        sa.Column('hash', sa.String(length=32), primary_key=True),
        sa.Column('model', sa.String(length=64), nullable=False),
        sa.Column('vec', HALFVEC(1024), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('claim_vector')
    op.drop_index('ix_cluster_member_key', table_name='cluster')
    op.drop_column('cluster', 'judged_key')
    op.drop_column('cluster', 'member_key')
