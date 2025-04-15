import re
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import SpanKind
from opentelemetry.semconv.trace import SpanAttributes


class PostgresSpanRenamer(SpanProcessor):
    """Custom SpanProcessor to rename PostgreSQL spans based on query content."""

    def on_start(self, span, parent_context=None):
        # Not modifying span on start in this implementation
        pass

    def on_end(self, span: ReadableSpan):
        """Modify span name on end if it's a PostgreSQL client span."""
        # Check if it's a PostgreSQL client span
        if (
            span.kind == SpanKind.CLIENT
            and span.attributes.get(SpanAttributes.DB_SYSTEM) == "postgresql"
        ):
            query = span.attributes.get(SpanAttributes.DB_STATEMENT, "")
            if query:
                operation = query.split()[0].upper() if query.split() else "QUERY"
                # Attempt to extract table name (simple example)
                table_name = self._extract_table_name(query)
                new_name = f"PG {operation}"
                if table_name:
                    new_name += f" {table_name}"
                # Directly modify the internal span name (use with caution, might depend on SDK version)
                # A safer approach might involve creating a new span or adding attributes
                # if direct name modification isn't supported or stable.
                # However, for demonstration, we attempt to modify the name.
                # Check if the span object allows name modification directly.
                # In some versions/implementations, span might be immutable on_end.
                # This is a conceptual example; actual implementation might need adjustment.
                try:
                    # This line assumes the span object is mutable and has a 'name' attribute
                    # that can be set. This might not be true for all SDK versions.
                    # If this fails, consider logging or adding attributes instead.
                    span._name = new_name  # Attempting direct modification
                except AttributeError:
                    # Fallback or logging if direct modification is not possible
                    print(
                        f"Could not directly rename span {span.context.span_id}. Attributes might be immutable."
                    )
                    # As an alternative, you could add an attribute:
                    # span.set_attribute("custom.span.name", new_name)

    def _extract_table_name(self, query):
        """Extracts table name from common SQL operations (simple example)."""
        # Regex for FROM or JOIN clauses
        match = re.search(r"\bFROM\s+([\w.]+)", query, re.IGNORECASE)
        if match:
            return match.group(1)

        # Regex for INSERT INTO
        match = re.search(r"\bINSERT\s+INTO\s+([\w.]+)", query, re.IGNORECASE)
        if match:
            return match.group(1)

        # Regex for UPDATE
        match = re.search(r"\bUPDATE\s+([\w.]+)", query, re.IGNORECASE)
        if match:
            return match.group(1)

        # Regex for DELETE FROM
        match = re.search(r"\bDELETE\s+FROM\s+([\w.]+)", query, re.IGNORECASE)
        if match:
            return match.group(1)

        return None

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        pass
