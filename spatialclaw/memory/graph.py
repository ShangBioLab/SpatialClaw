# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false, reportOperatorIssue=false, reportReturnType=false

"""
Graph Service for SpatialClaw Memory System.

Graph-based memory storage with:
- Node: a conceptual entity (UUID), version-independent
- Memory: a content version of a node
- Edge: parent->child relationship between nodes, carrying metadata
- Path: materialized URI cache (domain://path -> edge)

All infrastructure (engine, session, migrations) lives in database.py.
This module contains only graph-domain business logic.
"""

import uuid as uuid_lib
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from sqlalchemy import (
    select,
    update,
    delete,
    func,
    and_,
    or_,
    not_,
)
from sqlalchemy.ext.asyncio import AsyncSession
from .models import (
    ROOT_NODE_UUID,
    Node,
    Memory,
    Edge,
    Path,
    GlossaryKeyword,
    ChangeCollector,
    escape_like_literal,
    serialize_row,
    serialize_memory_ref,
)

if TYPE_CHECKING:
    from .database import DatabaseManager
    from .search import SearchIndexer


class GraphService:
    """
    Graph-domain service for memory operations.

    Owns graph traversal, memory CRUD, and path management. Receives a
    DatabaseManager for session access and a SearchIndexer for post-mutation
    index refreshes.

    Core operations:
    - read: Get memory by path (Path -> Edge -> Memory via node_uuid)
    - create: New node + memory + edge + path
    - update: New memory version on same node; update edge metadata
    - add_path: Create alias (new Path, maybe new Edge)
    - remove_path: Delete paths; refuse if children would become unreachable
    """

    def __init__(self, db: "DatabaseManager", search: "SearchIndexer"):
        self.session = db.session
        self._optional_session = db._optional_session
        self._search = search

    # =========================================================================
    # Read Operations
    # =========================================================================

    async def get_memory_by_path(
        self, path: str, domain: str = "core"
    ) -> Optional[Dict[str, Any]]:
        """
        Get a memory by its path.

        Returns:
            Memory dict with id, node_uuid, content, priority, disclosure,
            created_at, domain, path -- or None if not found.
        """
        if path == "":
            return {
                "id": 0,
                "node_uuid": ROOT_NODE_UUID,
                "content": f"Root node for domain '{domain}'.",
                "priority": 0,
                "disclosure": None,
                "created_at": None,
                "domain": domain,
                "path": "",
                "alias_count": 0,
            }

        async with self.session() as session:
            result = await session.execute(
                select(Memory, Edge, Path)
                .select_from(Path)
                .join(Edge, Path.edge_id == Edge.id)
                .join(
                    Memory,
                    and_(
                        Memory.node_uuid == Edge.child_uuid,
                        Memory.deprecated == False,
                    ),
                )
                .where(Path.domain == domain)
                .where(Path.path == path)
                .order_by(Memory.created_at.desc())
                .limit(1)
            )
            row = result.first()

            if not row:
                return None

            memory, edge, path_obj = row

            # Count total paths (aliases) for this node
            total_paths = await self._count_incoming_paths(session, edge.child_uuid)
            alias_count = max(0, total_paths - 1)

            return {
                "id": memory.id,
                "node_uuid": edge.child_uuid,
                "content": memory.content,
                "priority": edge.priority,
                "disclosure": edge.disclosure,
                "created_at": memory.created_at.isoformat()
                if memory.created_at
                else None,
                "domain": path_obj.domain,
                "path": path_obj.path,
                "alias_count": alias_count,
            }

    async def get_memory_by_node_uuid(self, node_uuid: str) -> Optional[Dict[str, Any]]:
        """Get the current active (non-deprecated) memory for a node."""
        async with self.session() as session:
            result = await session.execute(
                select(Memory)
                .where(Memory.node_uuid == node_uuid, Memory.deprecated == False)
                .order_by(Memory.created_at.desc())
                .limit(1)
            )
            memory = result.scalar_one_or_none()

            if not memory:
                return None

            paths_result = await session.execute(
                select(Path.domain, Path.path)
                .select_from(Path)
                .join(Edge, Path.edge_id == Edge.id)
                .where(Edge.child_uuid == node_uuid)
            )
            paths = [f"{r[0]}://{r[1]}" for r in paths_result.all()]

            return {
                "id": memory.id,
                "node_uuid": node_uuid,
                "content": memory.content,
                "created_at": memory.created_at.isoformat()
                if memory.created_at
                else None,
                "paths": paths,
            }

    async def get_children(
        self,
        node_uuid: str = ROOT_NODE_UUID,
        context_domain: Optional[str] = None,
        context_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get direct children of a node via the edges table.

        When *context_domain* / *context_path* are supplied the returned
        ``path`` for each child is chosen with affinity:
          1. Same domain AND path starts with ``context_path/``
          2. Same domain (any path)
          3. Any path at all
        """
        async with self.session() as session:
            stmt = (
                select(Edge, Memory)
                .join(
                    Memory,
                    and_(
                        Memory.node_uuid == Edge.child_uuid,
                        Memory.deprecated == False,
                    ),
                )
                .where(Edge.parent_uuid == node_uuid)
                .order_by(Edge.priority.asc(), Edge.name)
            )
            result = await session.execute(stmt)
            rows = result.all()

            prefix = f"{context_path}/" if context_path else None

            child_uuids = {edge.child_uuid for edge, _ in rows}
            approx_children_count_map: Dict[str, int] = {}
            if child_uuids:
                count_result = await session.execute(
                    select(Edge.parent_uuid, func.count(Edge.id))
                    .where(Edge.parent_uuid.in_(child_uuids))
                    .group_by(Edge.parent_uuid)
                )
                approx_children_count_map = {
                    parent_uuid: count for parent_uuid, count in count_result.all()
                }

            children = []
            seen = set()
            for edge, memory in rows:
                if edge.child_uuid in seen:
                    continue
                seen.add(edge.child_uuid)

                path_result = await session.execute(
                    select(Path).where(Path.edge_id == edge.id)
                )
                all_paths = path_result.scalars().all()

                if node_uuid == ROOT_NODE_UUID and context_domain:
                    has_domain_path = any(p.domain == context_domain for p in all_paths)
                    if not has_domain_path:
                        continue

                path_obj = self._pick_best_path(all_paths, context_domain, prefix)

                approx_children_count = approx_children_count_map.get(
                    edge.child_uuid, 0
                )

                children.append(
                    {
                        "node_uuid": edge.child_uuid,
                        "edge_id": edge.id,
                        "name": edge.name,
                        "domain": path_obj.domain if path_obj else "core",
                        "path": path_obj.path if path_obj else edge.name,
                        "content_snippet": memory.content[:100] + "..."
                        if len(memory.content) > 100
                        else memory.content,
                        "priority": edge.priority,
                        "disclosure": edge.disclosure,
                        "approx_children_count": approx_children_count,
                    }
                )

            return children

    @staticmethod
    def _pick_best_path(
        paths: List[Path],
        context_domain: Optional[str],
        prefix: Optional[str],
    ) -> Optional[Path]:
        """Pick the most contextually relevant path from a list of aliases."""
        if not paths:
            return None
        if len(paths) == 1:
            return paths[0]

        # Tier 1: same domain + path is under the caller's current prefix
        if context_domain and prefix:
            for p in paths:
                if p.domain == context_domain and p.path.startswith(prefix):
                    return p

        # Tier 2: same domain, any path
        if context_domain:
            for p in paths:
                if p.domain == context_domain:
                    return p

        # Tier 3: whatever is available
        return paths[0]

    async def get_all_paths(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all paths with their node/edge info."""
        async with self.session() as session:
            stmt = (
                select(Path, Edge, Memory)
                .select_from(Path)
                .join(Edge, Path.edge_id == Edge.id)
                .join(
                    Memory,
                    and_(
                        Memory.node_uuid == Edge.child_uuid,
                        Memory.deprecated == False,
                    ),
                )
            )

            if domain is not None:
                stmt = stmt.where(Path.domain == domain)

            stmt = stmt.order_by(Path.domain, Path.path)
            result = await session.execute(stmt)

            paths = []
            seen = set()
            for path_obj, edge, memory in result.all():
                key = (path_obj.domain, path_obj.path)
                if key in seen:
                    continue
                seen.add(key)
                paths.append(
                    {
                        "domain": path_obj.domain,
                        "path": path_obj.path,
                        "uri": f"{path_obj.domain}://{path_obj.path}",
                        "name": path_obj.path.rsplit("/", 1)[-1],
                        "priority": edge.priority,
                        "memory_id": memory.id,
                        "node_uuid": edge.child_uuid,
                    }
                )

            return paths

    # =========================================================================
    # Layer 0: Row-Level Primitives
    # =========================================================================

    async def _ensure_node(self, session: AsyncSession, node_uuid: str) -> Node:
        """Create a node if it doesn't exist; return it either way."""
        result = await session.execute(select(Node).where(Node.uuid == node_uuid))
        node = result.scalar_one_or_none()
        if node:
            return node
        node = Node(uuid=node_uuid)
        session.add(node)
        await session.flush()
        return node

    async def _insert_memory(
        self,
        session: AsyncSession,
        node_uuid: str,
        content: str,
        *,
        deprecated: bool = False,
    ) -> Memory:
        """Insert a new memory row and flush to obtain its ID."""
        memory = Memory(
            content=content,
            node_uuid=node_uuid,
            deprecated=deprecated,
        )
        session.add(memory)
        await session.flush()
        return memory

    async def _get_or_create_edge(
        self,
        session: AsyncSession,
        parent_uuid: str,
        child_uuid: str,
        name: str,
        priority: int = 0,
        disclosure: Optional[str] = None,
    ) -> tuple:
        """Get an existing edge or create a new one. Returns (edge, created: bool)."""
        result = await session.execute(
            select(Edge).where(
                Edge.parent_uuid == parent_uuid,
                Edge.child_uuid == child_uuid,
            )
        )
        edge = result.scalar_one_or_none()
        if edge:
            return edge, False

        edge = Edge(
            parent_uuid=parent_uuid,
            child_uuid=child_uuid,
            name=name,
            priority=priority,
            disclosure=disclosure,
        )
        session.add(edge)
        await session.flush()
        return edge, True

    async def _insert_path(
        self, session: AsyncSession, domain: str, path: str, edge_id: int
    ) -> Path:
        """Insert a new path row."""
        path_obj = Path(domain=domain, path=path, edge_id=edge_id)
        session.add(path_obj)
        return path_obj

    async def _resolve_path(
        self, session: AsyncSession, path: str, domain: str = "core"
    ) -> Optional[tuple]:
        """Resolve domain+path to (Path, Edge, node_uuid). Returns None if not found."""
        result = await session.execute(
            select(Path, Edge)
            .join(Edge, Path.edge_id == Edge.id)
            .where(Path.domain == domain, Path.path == path)
        )
        row = result.first()
        if not row:
            return None
        path_obj, edge = row
        return path_obj, edge, edge.child_uuid

    async def _count_paths_for_edge(self, session: AsyncSession, edge_id: int) -> int:
        """Count how many path rows reference a given edge."""
        result = await session.execute(
            select(func.count()).select_from(Path).where(Path.edge_id == edge_id)
        )
        return result.scalar()

    async def _count_incoming_paths(
        self,
        session: AsyncSession,
        node_uuid: str,
        *,
        exclude_domain: Optional[str] = None,
        exclude_path_prefix: Optional[str] = None,
    ) -> int:
        """Count paths whose edge points TO this node (edge.child_uuid)."""
        stmt = (
            select(func.count())
            .select_from(Path)
            .join(Edge, Path.edge_id == Edge.id)
            .where(Edge.child_uuid == node_uuid)
        )

        if exclude_domain and exclude_path_prefix:
            safe_prefix = escape_like_literal(exclude_path_prefix)
            stmt = stmt.where(
                not_(
                    and_(
                        Path.domain == exclude_domain,
                        Path.path.like(f"{safe_prefix}/%", escape="\\"),
                    )
                )
            )

        result = await session.execute(stmt)
        return result.scalar()

    async def _count_memories_for_node(
        self, session: AsyncSession, node_uuid: str
    ) -> int:
        """Count all memory rows (including deprecated) for a node."""
        result = await session.execute(
            select(func.count())
            .select_from(Memory)
            .where(Memory.node_uuid == node_uuid)
        )
        return result.scalar()

    async def _get_next_child_number(
        self, session: AsyncSession, parent_uuid: str
    ) -> int:
        """Get the next numeric name for auto-naming under a parent node."""
        result = await session.execute(
            select(Edge.name).where(Edge.parent_uuid == parent_uuid)
        )
        max_num = 0
        for (name,) in result.all():
            try:
                num = int(name)
                max_num = max(max_num, num)
            except ValueError:
                pass
        return max_num + 1

    async def _would_create_cycle(
        self,
        session: AsyncSession,
        parent_uuid: str,
        child_uuid: str,
    ) -> bool:
        """Check if adding edge parent_uuid->child_uuid would create a cycle."""
        if parent_uuid == ROOT_NODE_UUID:
            return False
        if parent_uuid == child_uuid:
            return True

        visited = {child_uuid}
        queue = [child_uuid]
        while queue:
            current = queue.pop(0)
            result = await session.execute(
                select(Edge.child_uuid).where(Edge.parent_uuid == current)
            )
            for (next_uuid,) in result.all():
                if next_uuid == parent_uuid:
                    return True
                if next_uuid not in visited:
                    visited.add(next_uuid)
                    queue.append(next_uuid)
        return False

    # =========================================================================
    # Layer 1: Table-Scoped Operations
    # =========================================================================

    async def _deprecate_node_memories(
        self,
        session: AsyncSession,
        node_uuid: str,
        *,
        successor_id: Optional[int] = None,
    ) -> List[int]:
        """Mark active memories for a node as deprecated."""
        conditions = [
            Memory.node_uuid == node_uuid,
            Memory.deprecated == False,
        ]
        if successor_id is not None:
            conditions.append(Memory.id != successor_id)

        result = await session.execute(select(Memory.id).where(and_(*conditions)))
        ids = [row[0] for row in result.all()]

        if ids:
            await session.execute(
                update(Memory)
                .where(Memory.id.in_(ids))
                .values(deprecated=True, migrated_to=successor_id)
            )
        return ids

    async def _get_subtree_path_rows(
        self,
        session: AsyncSession,
        domain: str,
        base_path: str,
    ) -> List[Dict[str, Any]]:
        """Return serialized path rows for base_path and all descendants."""
        safe = escape_like_literal(base_path)
        result = await session.execute(
            select(Path).where(
                Path.domain == domain,
                or_(
                    Path.path == base_path,
                    Path.path.like(f"{safe}/%", escape="\\"),
                ),
            )
        )
        return [serialize_row(p) for p in result.scalars().all()]

    async def _cascade_create_paths(
        self,
        session: AsyncSession,
        node_uuid: str,
        domain: str,
        base_path: str,
        _visited: Optional[set] = None,
    ):
        """Recursively create path entries for all descendants of a node."""
        if _visited is None:
            _visited = set()
        if node_uuid in _visited:
            return
        _visited.add(node_uuid)
        try:
            result = await session.execute(
                select(Edge).where(Edge.parent_uuid == node_uuid)
            )
            child_edges = result.scalars().all()

            for child_edge in child_edges:
                child_path = f"{base_path}/{child_edge.name}"

                existing = await session.execute(
                    select(Path)
                    .where(Path.domain == domain)
                    .where(Path.path == child_path)
                )
                if not existing.scalar_one_or_none():
                    session.add(
                        Path(domain=domain, path=child_path, edge_id=child_edge.id)
                    )

                await self._cascade_create_paths(
                    session, child_edge.child_uuid, domain, child_path, _visited
                )
        finally:
            _visited.remove(node_uuid)

    # =========================================================================
    # Layer 2: Cross-Table Cascades
    # =========================================================================

    async def _delete_subtree_paths(
        self,
        session: AsyncSession,
        domain: str,
        path: str,
        *,
        collector: Optional[ChangeCollector] = None,
    ) -> None:
        """Delete a path and all its descendant paths in the given domain."""
        safe = escape_like_literal(path)
        result = await session.execute(
            select(Path)
            .where(Path.domain == domain)
            .where(
                or_(
                    Path.path == path,
                    Path.path.like(f"{safe}/%", escape="\\"),
                )
            )
        )
        paths = result.scalars().all()

        for p in paths:
            serialized = serialize_row(p)
            if collector:
                collector.record("paths", serialized)
            await session.delete(p)

    async def _cascade_delete_edge(
        self,
        session: AsyncSession,
        edge: Edge,
        *,
        collector: Optional[ChangeCollector] = None,
    ) -> None:
        """Delete an edge, all its path references, and descendant paths."""
        paths_result = await session.execute(
            select(Path).where(Path.edge_id == edge.id)
        )
        edge_paths = paths_result.scalars().all()

        for p in edge_paths:
            await self._delete_subtree_paths(
                session,
                p.domain,
                p.path,
                collector=collector,
            )

        if collector:
            collector.record("edges", serialize_row(edge))
        await session.delete(edge)

    async def cascade_delete_node(
        self, session: AsyncSession, node_uuid: str
    ) -> Optional[Dict[str, list]]:
        """Hard-delete a node, all its memories, edges, and paths."""
        if node_uuid == ROOT_NODE_UUID:
            return None

        collector = ChangeCollector()

        edges_result = await session.execute(
            select(Edge).where(
                or_(Edge.parent_uuid == node_uuid, Edge.child_uuid == node_uuid)
            )
        )
        for edge in edges_result.scalars().all():
            await self._cascade_delete_edge(
                session,
                edge,
                collector=collector,
            )

        mem_result = await session.execute(
            select(Memory).where(Memory.node_uuid == node_uuid)
        )
        for mem in mem_result.scalars().all():
            collector.record("memories", serialize_row(mem))

        kw_result = await session.execute(
            select(GlossaryKeyword).where(GlossaryKeyword.node_uuid == node_uuid)
        )
        for kw in kw_result.scalars().all():
            collector.record("glossary_keywords", serialize_row(kw))

        await session.execute(
            delete(Memory).where(Memory.node_uuid == node_uuid)
        )
        node_row = await session.execute(select(Node).where(Node.uuid == node_uuid))
        node = node_row.scalar_one_or_none()
        if node:
            collector.record("nodes", serialize_row(node))
        await session.execute(delete(Node).where(Node.uuid == node_uuid))

        return collector.to_dict()

    async def _create_edge_with_paths(
        self,
        session: AsyncSession,
        parent_uuid: str,
        child_uuid: str,
        name: str,
        domain: str,
        path: str,
        priority: int = 0,
        disclosure: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create (or get) an edge, its path entry, and cascade sub-paths."""
        edge, edge_created = await self._get_or_create_edge(
            session, parent_uuid, child_uuid, name, priority, disclosure
        )
        path_obj = await self._insert_path(session, domain, path, edge.id)
        await self._cascade_create_paths(session, child_uuid, domain, path)
        return {
            "edge": edge,
            "edge_id": edge.id,
            "edge_created": edge_created,
            "path": path_obj,
        }

    # =========================================================================
    # Layer 3: GC / Conditional Logic
    # =========================================================================

    async def _gc_edge_if_pathless(
        self,
        session: AsyncSession,
        edge: Edge,
        *,
        collector: Optional[ChangeCollector] = None,
    ) -> Optional[Dict[str, Any]]:
        """Delete an edge only if it has no remaining path references."""
        if await self._count_paths_for_edge(session, edge.id) > 0:
            return None
        if collector:
            collector.record("edges", serialize_row(edge))
        info = {
            "edge_id": edge.id,
            "parent_uuid": edge.parent_uuid,
            "child_uuid": edge.child_uuid,
            "name": edge.name,
            "priority": edge.priority,
            "disclosure": edge.disclosure,
        }
        await session.delete(edge)
        return info

    async def _gc_node_soft(
        self,
        session: AsyncSession,
        node_uuid: str,
        *,
        collector: Optional[ChangeCollector] = None,
    ) -> None:
        """Soft GC: if a node has no incoming paths, deprecate its memories
        and cascade-delete all edges/paths around it."""
        if await self._count_incoming_paths(session, node_uuid) > 0:
            return

        incoming = await session.execute(
            select(Edge).where(Edge.child_uuid == node_uuid)
        )
        for edge in incoming.scalars().all():
            await self._gc_edge_if_pathless(session, edge, collector=collector)

        outgoing = await session.execute(
            select(Edge).where(Edge.parent_uuid == node_uuid)
        )
        for edge in outgoing.scalars().all():
            await self._cascade_delete_edge(
                session,
                edge,
                collector=collector,
            )

        if collector:
            active_mems = await session.execute(
                select(Memory).where(
                    Memory.node_uuid == node_uuid,
                    Memory.deprecated == False,
                )
            )
            for mem in active_mems.scalars().all():
                collector.record("memories", serialize_row(mem))

        await self._deprecate_node_memories(session, node_uuid)

    async def _gc_node_if_memoryless(
        self, session: AsyncSession, node_uuid: str
    ) -> Optional[Dict[str, list]]:
        """Hard GC: if a node has zero memory rows, cascade-delete everything."""
        if await self._count_memories_for_node(session, node_uuid) > 0:
            return None
        return await self.cascade_delete_node(session, node_uuid)

    # =========================================================================
    # Public Write API
    # =========================================================================

    async def create_memory(
        self,
        parent_path: str,
        content: str,
        priority: int,
        title: Optional[str] = None,
        disclosure: Optional[str] = None,
        domain: str = "core",
    ) -> Dict[str, Any]:
        """
        Create a new memory under a parent path.

        Creates: Node -> Memory -> Edge (parent->child) -> Path.
        """
        async with self.session() as session:
            if not parent_path:
                parent_uuid = ROOT_NODE_UUID
            else:
                parent = await self._resolve_path(session, parent_path, domain)
                if not parent:
                    raise ValueError(
                        f"Parent '{domain}://{parent_path}' does not exist. "
                        f"Create the parent first, or use '{domain}://' as root."
                    )
                _, _, parent_uuid = parent

            if title:
                final_path = f"{parent_path}/{title}" if parent_path else title
            else:
                next_num = await self._get_next_child_number(session, parent_uuid)
                final_path = (
                    f"{parent_path}/{next_num}" if parent_path else str(next_num)
                )

            existing = await session.execute(
                select(Path).where(Path.domain == domain, Path.path == final_path)
            )
            if existing.scalar_one_or_none():
                raise ValueError(f"Path '{domain}://{final_path}' already exists")

            new_uuid = str(uuid_lib.uuid4())
            node = await self._ensure_node(session, new_uuid)
            memory = await self._insert_memory(session, new_uuid, content)

            edge_name = title if title else final_path.rsplit("/", 1)[-1]
            created = await self._create_edge_with_paths(
                session,
                parent_uuid,
                new_uuid,
                edge_name,
                domain,
                final_path,
                priority,
                disclosure,
            )

            await self._search.refresh_search_documents_for_node(new_uuid, session=session)

            return {
                "id": memory.id,
                "node_uuid": new_uuid,
                "domain": domain,
                "path": final_path,
                "uri": f"{domain}://{final_path}",
                "priority": priority,
                "rows_after": {
                    "nodes": [serialize_row(node)],
                    "memories": [serialize_memory_ref(memory)],
                    "edges": [serialize_row(created["edge"])],
                    "paths": [serialize_row(created["path"])],
                },
            }

    async def update_memory(
        self,
        path: str,
        content: Optional[str] = None,
        priority: Optional[int] = None,
        disclosure: Optional[str] = None,
        domain: str = "core",
    ) -> Dict[str, Any]:
        """
        Update a memory.

        Content change -> new Memory row with the same node_uuid.
        Metadata change -> update the Edge directly.
        """
        if path == "":
            raise ValueError("Cannot update the root node.")

        if content is None and priority is None and disclosure is None:
            raise ValueError(
                f"No update fields provided for '{domain}://{path}'. "
                "At least one of content, priority, or disclosure must be set."
            )

        async with self.session() as session:
            result = await session.execute(
                select(Memory, Edge, Path)
                .select_from(Path)
                .join(Edge, Path.edge_id == Edge.id)
                .join(
                    Memory,
                    and_(
                        Memory.node_uuid == Edge.child_uuid,
                        Memory.deprecated == False,
                    ),
                )
                .where(Path.domain == domain, Path.path == path)
                .order_by(Memory.created_at.desc())
                .limit(1)
            )
            row = result.first()

            if not row:
                raise ValueError(
                    f"Path '{domain}://{path}' not found or memory is deprecated"
                )

            old_memory, edge, path_obj = row
            old_id = old_memory.id
            node_uuid = edge.child_uuid

            rows_before: Dict[str, list] = {}
            rows_after: Dict[str, list] = {}

            edge_before = serialize_row(edge)

            if priority is not None:
                edge.priority = priority
                session.add(edge)
            if disclosure is not None:
                edge.disclosure = disclosure
                session.add(edge)

            edge_after = serialize_row(edge)
            if edge_before != edge_after:
                rows_before["edges"] = [edge_before]
                rows_after["edges"] = [edge_after]

            new_memory_id = old_id

            if content is not None:
                rows_before["memories"] = [serialize_memory_ref(old_memory)]

                new_memory = await self._insert_memory(
                    session, node_uuid, content, deprecated=True
                )
                new_memory_id = new_memory.id
                await self._deprecate_node_memories(
                    session,
                    node_uuid,
                    successor_id=new_memory_id,
                )
                await session.execute(
                    update(Memory)
                    .where(Memory.id == new_memory_id)
                    .values(deprecated=False, migrated_to=None)
                )

                await session.flush()
                updated = await session.execute(
                    select(Memory).where(Memory.id.in_([old_id, new_memory_id]))
                )
                rows_after["memories"] = [
                    serialize_memory_ref(m) for m in updated.scalars().all()
                ]

            if content is None:
                session.add(path_obj)

            await self._search.refresh_search_documents_for_node(node_uuid, session=session)

            return {
                "domain": domain,
                "path": path,
                "uri": f"{domain}://{path}",
                "old_memory_id": old_id,
                "new_memory_id": new_memory_id,
                "node_uuid": node_uuid,
                "rows_before": rows_before,
                "rows_after": rows_after,
            }

    async def add_path(
        self,
        new_path: str,
        target_path: str,
        new_domain: str = "core",
        target_domain: str = "core",
        priority: int = 0,
        disclosure: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an alias path pointing to the same node as target_path."""
        if new_path == "":
            raise ValueError("Cannot create an alias at the root path.")

        async with self.session() as session:
            target = await self._resolve_path(session, target_path, target_domain)
            if not target:
                raise ValueError(
                    f"Target path '{target_domain}://{target_path}' not found"
                )
            _, _, target_node_uuid = target

            if "/" in new_path:
                parent_path = new_path.rsplit("/", 1)[0]
                parent = await self._resolve_path(session, parent_path, new_domain)
                if not parent:
                    raise ValueError(
                        f"Parent '{new_domain}://{parent_path}' does not exist. "
                        f"Create the parent first, or use a shallower alias path."
                    )
                _, _, parent_uuid = parent
            else:
                parent_uuid = ROOT_NODE_UUID

            existing = await session.execute(
                select(Path).where(Path.domain == new_domain, Path.path == new_path)
            )
            if existing.scalar_one_or_none():
                raise ValueError(f"Path '{new_domain}://{new_path}' already exists")

            before_subtree = await self._get_subtree_path_rows(
                session, new_domain, new_path
            )
            before_path_keys = {(row["domain"], row["path"]) for row in before_subtree}

            if await self._would_create_cycle(session, parent_uuid, target_node_uuid):
                raise ValueError(
                    f"Cannot create alias '{new_domain}://{new_path}': "
                    f"target node is an ancestor of the destination parent, "
                    f"which would create a cycle in the graph."
                )

            result = await self._create_edge_with_paths(
                session,
                parent_uuid,
                target_node_uuid,
                new_path.rsplit("/", 1)[-1],
                new_domain,
                new_path,
                priority,
                disclosure,
            )
            await session.flush()

            after_subtree = await self._get_subtree_path_rows(
                session, new_domain, new_path
            )
            created_paths = [
                row
                for row in after_subtree
                if (row["domain"], row["path"]) not in before_path_keys
            ]

            rows_after: Dict[str, list] = {
                "paths": created_paths,
            }
            if result["edge_created"]:
                rows_after["edges"] = [serialize_row(result["edge"])]

            affected_nodes = await self._search.get_node_uuids_for_prefix(session, new_domain, new_path)
            for node_uuid in affected_nodes:
                await self._search.refresh_search_documents_for_node(node_uuid, session=session)

            return {
                "new_uri": f"{new_domain}://{new_path}",
                "target_uri": f"{target_domain}://{target_path}",
                "node_uuid": target_node_uuid,
                "edge_id": result["edge_id"],
                "edge_created": result["edge_created"],
                "rows_after": rows_after,
            }

    async def remove_path(
        self, path: str, domain: str = "core", session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Remove a path and its sub-paths with orphan prevention."""
        if path == "":
            raise ValueError("Cannot remove the root path.")

        async with self._optional_session(session) as session:
            target = await self._resolve_path(session, path, domain)
            if not target:
                raise ValueError(f"Path '{domain}://{path}' not found")
            _, target_edge, target_node_uuid = target

            # Pre-flight orphan check
            child_edges_result = await session.execute(
                select(Edge).where(Edge.parent_uuid == target_node_uuid)
            )
            child_edges = child_edges_result.scalars().all()

            would_orphan = []
            for child_edge in child_edges:
                surviving_count = await self._count_incoming_paths(
                    session,
                    child_edge.child_uuid,
                    exclude_domain=domain,
                    exclude_path_prefix=path,
                )
                if surviving_count == 0:
                    would_orphan.append(child_edge)

            if would_orphan:
                details = ", ".join(
                    f"'{e.name}' (node: {e.child_uuid[:8]}...)" for e in would_orphan
                )
                raise ValueError(
                    f"Cannot remove '{domain}://{path}': "
                    f"the following child node(s) would become unreachable: "
                    f"{details}. "
                    f"Create alternative paths for these children first, "
                    f"or remove them explicitly."
                )

            collector = ChangeCollector()
            affected_nodes = await self._search.get_node_uuids_for_prefix(session, domain, path)
            await self._delete_subtree_paths(session, domain, path, collector=collector)
            await session.flush()

            await self._gc_edge_if_pathless(session, target_edge, collector=collector)
            await self._gc_node_soft(session, target_node_uuid, collector=collector)

            for node_uuid in affected_nodes:
                await self._search.refresh_search_documents_for_node(node_uuid, session=session)

            return {
                "rows_before": collector.to_dict(),
                "rows_after": {},
            }

    # =========================================================================
    # Recent Memories
    # =========================================================================

    async def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recently created/updated non-deprecated memories."""
        async with self.session() as session:
            result = await session.execute(
                select(Memory, Edge, Path)
                .select_from(Path)
                .join(Edge, Path.edge_id == Edge.id)
                .join(
                    Memory,
                    and_(
                        Memory.node_uuid == Edge.child_uuid,
                        Memory.deprecated == False,
                    ),
                )
                .order_by(Memory.created_at.desc())
            )

            seen = set()
            memories = []

            for memory, edge, path_obj in result.all():
                if memory.id in seen:
                    continue
                seen.add(memory.id)

                memories.append(
                    {
                        "memory_id": memory.id,
                        "uri": f"{path_obj.domain}://{path_obj.path}",
                        "priority": edge.priority,
                        "disclosure": edge.disclosure,
                        "created_at": memory.created_at.isoformat()
                        if memory.created_at
                        else None,
                    }
                )

                if len(memories) >= limit:
                    break

            return memories

    # =========================================================================
    # Deprecated Memory Operations (for human review)
    # =========================================================================

    async def get_memory_by_id(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Get a memory by its ID (including deprecated ones)."""
        async with self.session() as session:
            result = await session.execute(select(Memory).where(Memory.id == memory_id))
            memory = result.scalar_one_or_none()

            if not memory:
                return None

            paths = []
            if memory.node_uuid:
                paths_result = await session.execute(
                    select(Path.domain, Path.path)
                    .select_from(Path)
                    .join(Edge, Path.edge_id == Edge.id)
                    .where(Edge.child_uuid == memory.node_uuid)
                )
                paths = [f"{r[0]}://{r[1]}" for r in paths_result.all()]

            return {
                "memory_id": memory.id,
                "node_uuid": memory.node_uuid,
                "content": memory.content,
                "created_at": memory.created_at.isoformat()
                if memory.created_at
                else None,
                "paths": paths,
            }

    # =========================================================================
    # Task Similarity Graph
    # =========================================================================

    async def add_task_similarity_edge(
        self,
        task_uri_a: str,
        task_uri_b: str,
        similarity: float,
    ) -> None:
        """
        Add a task-similarity edge between two AnalysisMemory nodes.

        Uses Edge.disclosure to store the similarity score (avoids schema change).
        Edge.name = "task_similar" as type marker.
        """
        domain_a, path_a = task_uri_a.split("://", 1)
        domain_b, path_b = task_uri_b.split("://", 1)

        async with self.session() as session:
            result_a = await self._resolve_path(session, path_a, domain_a)
            result_b = await self._resolve_path(session, path_b, domain_b)
            if not result_a or not result_b:
                return

            _, _, node_uuid_a = result_a
            _, _, node_uuid_b = result_b

            # Check if edge already exists (undirected: check both directions)
            existing = await session.execute(
                select(Edge).where(
                    or_(
                        and_(
                            Edge.parent_uuid == node_uuid_a,
                            Edge.child_uuid == node_uuid_b,
                            Edge.name == "task_similar",
                        ),
                        and_(
                            Edge.parent_uuid == node_uuid_b,
                            Edge.child_uuid == node_uuid_a,
                            Edge.name == "task_similar",
                        ),
                    )
                )
            )
            if existing.scalar_one_or_none():
                return

            edge = Edge(
                parent_uuid=node_uuid_a,
                child_uuid=node_uuid_b,
                name="task_similar",
                priority=0,
                disclosure=str(round(similarity, 4)),
            )
            session.add(edge)

    async def get_task_neighbors(
        self,
        task_uri: str,
        hop: int = 1,
        similarity_threshold: float = 0.0,
    ) -> list[str]:
        """
        Get neighbor task URIs within k hops via task_similar edges.

        Returns list of URIs (not including the query node itself).
        """
        domain, path = task_uri.split("://", 1)

        async with self.session() as session:
            resolved = await self._resolve_path(session, path, domain)
            if not resolved:
                return []
            _, _, start_uuid = resolved

            visited_uuids: set[str] = {start_uuid}
            frontier: set[str] = {start_uuid}

            for _ in range(hop):
                next_frontier: set[str] = set()
                if not frontier:
                    break

                result = await session.execute(
                    select(Edge).where(
                        and_(
                            Edge.name == "task_similar",
                            or_(
                                Edge.parent_uuid.in_(frontier),
                                Edge.child_uuid.in_(frontier),
                            ),
                        )
                    )
                )
                for edge in result.scalars().all():
                    if similarity_threshold > 0:
                        try:
                            sim = float(edge.disclosure or "0")
                            if sim < similarity_threshold:
                                continue
                        except ValueError:
                            pass

                    neighbor = (
                        edge.child_uuid
                        if edge.parent_uuid in frontier
                        else edge.parent_uuid
                    )
                    if neighbor not in visited_uuids:
                        next_frontier.add(neighbor)
                        visited_uuids.add(neighbor)

                frontier = next_frontier

            if not visited_uuids - {start_uuid}:
                return []

            neighbor_uris: list[str] = []
            for node_uuid in visited_uuids - {start_uuid}:
                paths_result = await session.execute(
                    select(Path.domain, Path.path)
                    .select_from(Path)
                    .join(Edge, Path.edge_id == Edge.id)
                    .where(Edge.child_uuid == node_uuid)
                    .limit(1)
                )
                row = paths_result.first()
                if row:
                    neighbor_uris.append(f"{row[0]}://{row[1]}")

            return neighbor_uris
        

    async def add_insight_task_edge(
        self,
        insight_uri: str,
        task_uri: str,
    ) -> None:
        """
        Add a directed edge from an InsightMemory node to an AnalysisMemory node.

        edge.name = "derived_from"
        """
        domain_i, path_i = insight_uri.split("://", 1)
        domain_t, path_t = task_uri.split("://", 1)

        async with self.session() as session:
            result_i = await self._resolve_path(session, path_i, domain_i)
            result_t = await self._resolve_path(session, path_t, domain_t)
            if not result_i or not result_t:
                return

            _, _, node_uuid_i = result_i
            _, _, node_uuid_t = result_t

            existing = await session.execute(
                select(Edge).where(
                    Edge.parent_uuid == node_uuid_i,
                    Edge.child_uuid == node_uuid_t,
                    Edge.name == "derived_from",
                )
            )
            if existing.scalar_one_or_none():
                return

            edge = Edge(
                parent_uuid=node_uuid_i,
                child_uuid=node_uuid_t,
                name="derived_from",
                priority=0,
                disclosure=None,
            )
            session.add(edge)
