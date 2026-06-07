"""Output rendering for the ``ss-query`` CLI."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import Console
from rich.table import Table

from .query import format_value


console = Console()


def render_records(records: Sequence[Mapping[str, Any]], *, fmt: str) -> None:
    """Render a list of mapping records."""
    if fmt == "json":
        console.print_json(json.dumps(list(records), default=str))
        return
    if fmt == "csv":
        _write_csv(records)
        return
    _render_table(records)


def render_dataframe(df, *, fmt: str) -> None:
    """Render a pandas dataframe."""
    if fmt == "json":
        console.print_json(df.to_json(orient="records", default_handler=str))
        return
    if fmt == "csv":
        df.to_csv(sys.stdout, index=False)
        return
    records = df.to_dict(orient="records")
    _render_table(records)


def _render_table(records: Sequence[Mapping[str, Any]]) -> None:
    table = Table(show_lines=False)
    columns = list(records[0].keys()) if records else ["result"]
    for column in columns:
        table.add_column(str(column))

    if records:
        for record in records:
            table.add_row(*(format_value(record.get(column)) for column in columns))
    else:
        table.add_row("")

    console.print(table)


def _write_csv(records: Sequence[Mapping[str, Any]]) -> None:
    columns = list(records[0].keys()) if records else ["result"]
    writer = csv.DictWriter(sys.stdout, fieldnames=columns)
    writer.writeheader()
    for record in records:
        writer.writerow({column: format_value(record.get(column)) for column in columns})

