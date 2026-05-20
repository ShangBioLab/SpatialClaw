"""Review API for pending memory changes."""

from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Query

from ..snapshot import get_changeset_store, _rows_equal

router = APIRouter(prefix="/api/review", tags=["review"])


def _public_row(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Hide internal graph version-chain fields from review responses."""
    if row is None:
        return None
    if not isinstance(row, dict):
        return row
    cleaned = dict(row)
    cleaned.pop("deprecated", None)
    cleaned.pop("migrated_to", None)
    return cleaned


def _public_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **entry,
        "before": _public_row(entry.get("before")),
        "after": _public_row(entry.get("after")),
    }


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.get("/changes")
async def get_changes():
    """Get all pending AI changes grouped by affected node."""
    store = get_changeset_store()

    # Use get_all_rows_dict to get key→entry mapping
    all_rows = store.get_all_rows_dict()
    # Filter to only net-changed rows
    changed_keys = set()
    for key, entry in all_rows.items():
        table = entry.get("table", "")
        before = entry.get("before")
        after = entry.get("after")
        if not _rows_equal(table, before, after):
            changed_keys.add(key)

    change_count = len(changed_keys)

    # Group by node_uuid, including key in each entry
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for key in changed_keys:
        entry = all_rows[key]
        table = entry.get("table", "unknown")
        before = entry.get("before")
        after = entry.get("after")
        ref = after or before

        # Determine node_uuid from the row
        node_uuid = None
        if ref:
            if table == "nodes":
                node_uuid = ref.get("uuid")
            elif table == "memories":
                node_uuid = ref.get("node_uuid")
            elif table == "edges":
                node_uuid = ref.get("child_uuid")
            elif table == "paths":
                # For paths, we need to find node_uuid via the edge
                node_uuid = ref.get("node_uuid")
                if not node_uuid:
                    edge_id = ref.get("edge_id")
                    if edge_id:
                        edge_key = f"edges:{edge_id}"
                        edge_entry = all_rows.get(edge_key)
                        if edge_entry:
                            edge_ref = edge_entry.get("after") or edge_entry.get("before")
                            if edge_ref:
                                node_uuid = edge_ref.get("child_uuid")
            elif table == "glossary_keywords":
                node_uuid = ref.get("node_uuid")

        if not node_uuid:
            node_uuid = "_unknown_"

        if node_uuid not in groups:
            groups[node_uuid] = []

        # Include the key so the frontend can reference individual changes.
        entry_with_key = _public_entry({**entry, "key": key})
        groups[node_uuid].append(entry_with_key)

    return {
        "change_count": change_count,
        "groups": groups,
    }


@router.get("/change-count")
async def get_change_count():
    """Get the number of pending changes."""
    store = get_changeset_store()
    return {"count": store.get_change_count()}


@router.get("/diff")
async def get_diff(key: str = Query(..., description="Change key to diff")):
    """Get before/after diff for a specific change."""
    store = get_changeset_store()
    all_rows = store.get_all_rows_dict()

    entry = all_rows.get(key)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Change key '{key}' not found")

    table = entry.get("table", "unknown")
    before = entry.get("before")
    after = entry.get("after")

    # For memory rows, resolve content from DB
    if table == "memories":
        from .. import get_graph_service
        graph = get_graph_service()

        if before and "id" in before:
            mem = await graph.get_memory_by_id(before["id"])
            if mem:
                before = {**before, "content": mem.get("content", "")}

        if after and "id" in after:
            mem = await graph.get_memory_by_id(after["id"])
            if mem:
                after = {**after, "content": mem.get("content", "")}

    return {
        "key": key,
        "table": table,
        "before": _public_row(before),
        "after": _public_row(after),
    }


@router.post("/integrate-all")
async def integrate_all():
    """Accept all pending changes (clear the changeset)."""
    store = get_changeset_store()
    count = store.clear_all()
    return {"integrated": count}
