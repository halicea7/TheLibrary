"""or_tsquery helper

Revision ID: b1f3a9c0d2e4
Revises: 4e2ca732dafe
Create Date: 2026-09-18

"""

from collections.abc import Sequence

from alembic import op

revision: str = "b1f3a9c0d2e4"
down_revision: str | Sequence[str] | None = "4e2ca732dafe"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# websearch_to_tsquery / plainto_tsquery both AND every term, so a natural-language
# question like "how does removing NSP affect BERT performance" matches zero rows. BM25-
# style behaviour needs OR semantics with ts_rank_cd doing the discrimination.
_UP = """
create or replace function or_tsquery(q text)
returns tsquery
language sql
immutable
parallel safe
as $$
    select nullif(
        array_to_string(
            array(select unnest(tsvector_to_array(to_tsvector('english', coalesce(q, ''))))),
            ' | '
        ),
        ''
    )::tsquery
$$;
"""


def upgrade() -> None:
    op.execute(_UP)


def downgrade() -> None:
    op.execute("drop function if exists or_tsquery(text)")
