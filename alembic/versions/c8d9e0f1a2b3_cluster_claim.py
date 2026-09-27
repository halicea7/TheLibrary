"""Threads hold claims: one row per claim in a cluster

Revision ID: c8d9e0f1a2b3
Revises: b7c8d9e0f1a2
Create Date: 2026-09-27 18:30:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c8d9e0f1a2b3'
down_revision: Union[str, Sequence[str], None] = 'b7c8d9e0f1a2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # A thread's membership, claim by claim: which Tier 1 claim (the section summary that
    # holds it, and its place in that summary's list), its words, and its volume. The
    # section-level cluster_member rows stay for the graph and the shelf.
    op.create_table(
        'cluster_claim',
        sa.Column('cluster_id', sa.Uuid(), sa.ForeignKey('cluster.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('artifact_id', sa.Uuid(), sa.ForeignKey('artifact.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('claim_index', sa.Integer(), primary_key=True),
        sa.Column('claim_hash', sa.String(length=32), nullable=False),
        sa.Column('document_id', sa.Uuid(), sa.ForeignKey('document.id', ondelete='CASCADE'), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
    )
    op.create_index('ix_cluster_claim_document_id', 'cluster_claim', ['document_id'])
    op.create_index('ix_cluster_claim_claim_hash', 'cluster_claim', ['claim_hash'])


def downgrade() -> None:
    op.drop_table('cluster_claim')
