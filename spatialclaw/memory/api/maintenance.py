"""Maintenance API for canonical memory operations."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/maintenance", tags=["maintenance"])


@router.post("/rebuild-search-index")
async def rebuild_search_index():
    """Fully rebuild the search index from live graph state."""
    from .. import get_search_indexer
    search = get_search_indexer()

    await search.rebuild_all_search_documents()
    return {"status": "ok", "message": "Search index rebuilt successfully"}
