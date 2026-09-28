"""Every Tier 1 claim tied to the passage in its section that supports it

Revision ID: e0f1a2b3c4d5
Revises: d9e0f1a2b3c4
Create Date: 2026-09-28 10:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e0f1a2b3c4d5'
down_revision: Union[str, Sequence[str], None] = 'd9e0f1a2b3c4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # A claim is (the section summary that holds it, its place in the list). Its anchor is
    # the passage of that section nearest it, how near, and what that nearness means:
    # "anchored", "weak" (partial support at best), "unsupported" (nothing in its own
    # section says it -- the reader may have overreached), "unembedded" (no vector yet).
    op.create_table(
        'claim_anchor',
        sa.Column('artifact_id', sa.Uuid(), sa.ForeignKey('artifact.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('claim_index', sa.Integer(), primary_key=True),
        sa.Column('claim_hash', sa.String(length=32), nullable=False),
        sa.Column('document_id', sa.Uuid(), sa.ForeignKey('document.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_id', sa.Uuid(), sa.ForeignKey('chunk.id', ondelete='CASCADE'), nullable=True),
        sa.Column('similarity', sa.Float(), nullable=True),
        sa.Column('state', sa.String(length=12), nullable=False),
    )
    op.create_index('ix_claim_anchor_claim_hash', 'claim_anchor', ['claim_hash'])
    op.create_index('ix_claim_anchor_document_id', 'claim_anchor', ['document_id'])
    op.create_index('ix_claim_anchor_state', 'claim_anchor', ['state'])


def downgrade() -> None:
    op.drop_table('claim_anchor')
