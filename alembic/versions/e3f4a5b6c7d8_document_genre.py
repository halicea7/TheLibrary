"""document.genre: what kind of writing this is, which changes how it is read

Revision ID: e3f4a5b6c7d8
Revises: d2e3f4a5b6c7
Create Date: 2026-09-20 21:10:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e3f4a5b6c7d8'
down_revision: Union[str, Sequence[str], None] = 'd2e3f4a5b6c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('document', sa.Column('genre', sa.String(length=24), nullable=True))
    op.add_column('cartridge', sa.Column('genre', sa.String(length=24), nullable=True))
    # What the orientation cards already decided, kept where the readers can see it.
    op.execute("""
        update document d set genre = case a.data->>'document_kind'
            when 'documentation' then 'documentation' when 'paper' then 'paper'
            when 'book' then 'book' when 'report' then 'report' when 'notes' then 'notes'
            else null end
        from artifact a where a.kind = 'orientation' and a.target_id = d.id and d.genre is null
    """)


def downgrade() -> None:
    op.drop_column('cartridge', 'genre')
    op.drop_column('document', 'genre')
