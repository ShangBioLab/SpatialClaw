import { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import { AlertTriangle, GitBranch, Link2, ListTree, Network } from 'lucide-react';

import { getAllGlossary, getAllPaths, getNode } from '../../../lib/memoryApi';
import EmptyState from '../../../shared/components/EmptyState';
import LoadingState from '../../../shared/components/LoadingState';
import StatusPill from '../../../shared/components/StatusPill';
import { formatDisplayText, formatMemoryUri, formatNodeTitle } from '../../../shared/utils/content';
import { buildLayeredMemoryGraph, buildMemoryGraph, shouldUseLayeredGraph } from './graphBuilder';

const kindClasses = {
  current: 'fill-cyan-400 stroke-cyan-200',
  child: 'fill-emerald-400 stroke-emerald-200',
  alias: 'fill-violet-400 stroke-violet-200',
  keyword: 'fill-amber-300 stroke-amber-100',
  sessionHub: 'fill-cyan-300 stroke-cyan-100',
  session: 'fill-cyan-400 stroke-cyan-200',
  episodic: 'fill-emerald-400 stroke-emerald-200',
  semantic: 'fill-violet-400 stroke-violet-200',
  memoryType: 'fill-amber-300 stroke-amber-100',
  memoryLeaf: 'fill-slate-300 stroke-slate-100',
};

const edgeClasses = {
  child: 'stroke-emerald-500/60',
  alias: 'stroke-violet-400/60',
  glossary: 'stroke-amber-400/60',
  session: 'stroke-cyan-400/60',
  episodic: 'stroke-emerald-400/60',
  semantic: 'stroke-violet-400/60',
  memoryType: 'stroke-amber-300/60',
  memoryLeaf: 'stroke-slate-400/60',
};

export default function GraphView({ node, children, domain, path, onNavigate }) {
  const [extra, setExtra] = useState({ allPaths: [], glossary: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedId, setSelectedId] = useState('');

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError('');
    const fallback = (request) => request.catch(() => []);
    Promise.all([
      fallback(getAllPaths().then((data) => data.paths || [])),
      fallback(getAllGlossary().then((data) => data.glossary || [])),
    ])
      .then(([allPaths, glossary]) => {
        if (!alive) return;
        setExtra({ allPaths, glossary });
      })
      .catch((err) => {
        if (!alive) return;
        setError(err.response?.data?.detail || err.message || 'Unable to load graph data');
      })
      .finally(() => {
        if (alive) setLoading(false);
      });

    return () => {
      alive = false;
    };
  }, [domain]);

  const graph = useMemo(() => {
    if (shouldUseLayeredGraph(domain)) {
      return buildLayeredMemoryGraph({ domain, path, allPaths: extra.allPaths });
    }
    return buildMemoryGraph({ node, children, domain, path, ...extra });
  }, [node, children, domain, path, extra]);

  const selected = graph.nodes.find((item) => item.id === selectedId) || graph.nodes[0];

  if (loading) {
    return (
      <div className="flex min-h-[520px] items-center justify-center rounded-lg border border-slate-200 bg-white">
        <LoadingState label="Building memory graph" />
      </div>
    );
  }

  if (error) {
    return <EmptyState icon={AlertTriangle} title="Graph unavailable" detail={error} />;
  }

  if (!graph.nodes.length) {
    return <EmptyState icon={Network} title="No graph data" detail="This node has no visible relationships from the current API." />;
  }

  if (graph.layered) {
    return (
      <MemoryTimeline
        domain={domain}
        path={path}
        allPaths={extra.allPaths}
        stats={graph.stats}
        onNavigate={onNavigate}
      />
    );
  }

  return (
    <div className="grid min-h-[620px] gap-4 xl:grid-cols-[1fr_320px]">
      <section className="surface overflow-hidden rounded-lg">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 p-4">
          <div>
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-950">
              <Network size={17} className="text-cyan-300" />
              {graph.layered ? 'Layered Memory Map' : 'Memory Relationship Map'}
            </div>
            <div className="mt-1 text-xs text-slate-500">
              {graph.layered
                ? 'Session, episodic, and semantic memory grouped by session id and memory type.'
                : 'Existing API view: hierarchy, aliases, and glossary links.'}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {graph.layered ? (
              <>
                <StatusPill tone="cyan">{graph.stats.sessions} sessions</StatusPill>
                <StatusPill tone="green">{graph.stats.episodic} episodic paths</StatusPill>
                <StatusPill>{graph.stats.semantic} semantic paths</StatusPill>
              </>
            ) : (
              <>
                <StatusPill tone="green">{graph.stats.children} children</StatusPill>
                <StatusPill>{graph.stats.aliases} aliases</StatusPill>
                <StatusPill tone="amber">{graph.stats.glossary} terms</StatusPill>
              </>
            )}
          </div>
        </div>
        <GraphCanvas graph={graph} selectedId={selected?.id} onSelect={setSelectedId} onNavigate={onNavigate} />
        <SelectionDetailPanel
          graph={graph}
          selected={selected}
          onSelect={setSelectedId}
          onNavigate={onNavigate}
        />
      </section>

      <aside className="surface rounded-lg p-4">
        <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-950">
          <GitBranch size={16} className="text-cyan-300" />
          Selected Relation
        </div>
        {selected ? <NodeInspector node={selected} onNavigate={onNavigate} /> : null}
        <div className="mt-5 space-y-2 text-xs text-slate-500">
          {graph.layered ? (
            <>
              <Legend color="bg-cyan-400" label="Session hub" />
              <Legend color="bg-emerald-400" label="Episodic layer" />
              <Legend color="bg-violet-400" label="Semantic layer" />
              <Legend color="bg-amber-300" label="Memory type group" />
              <Legend color="bg-slate-300" label="Concrete memory item" />
            </>
          ) : (
            <>
              <Legend color="bg-cyan-400" label="Current node" />
              <Legend color="bg-emerald-400" label="Child path" />
              <Legend color="bg-violet-400" label="Alias path" />
              <Legend color="bg-amber-300" label="Glossary keyword" />
            </>
          )}
        </div>
      </aside>
    </div>
  );
}

function GraphCanvas({ graph, selectedId, onSelect, onNavigate }) {
  const positioned = useMemo(() => positionNodes(graph.nodes), [graph.nodes]);
  const byId = new Map(positioned.map((item) => [item.id, item]));

  return (
    <div className="h-[560px] overflow-hidden bg-[radial-gradient(circle_at_50%_48%,rgba(14,165,233,0.10),transparent_34%),#09090b]">
      <svg viewBox="0 0 1000 620" className="h-full w-full" role="img" aria-label="Memory relationship graph">
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M 0 0 L 8 4 L 0 8 z" fill="#71717a" />
          </marker>
        </defs>
        {graph.edges.map((edge) => {
          const source = byId.get(edge.source);
          const target = byId.get(edge.target);
          if (!source || !target) return null;
          return (
            <g key={edge.id}>
              <line
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                markerEnd="url(#arrow)"
                className={clsx('stroke-[2]', edgeClasses[edge.kind] || 'stroke-slate-300')}
              />
              <text
                x={(source.x + target.x) / 2}
                y={(source.y + target.y) / 2 - 7}
                textAnchor="middle"
                className="select-none fill-slate-500 text-[11px]"
              >
                {edge.label}
              </text>
            </g>
          );
        })}
        {positioned.map((item) => (
          <GraphNode
            key={item.id}
            item={item}
            active={item.id === selectedId}
            onSelect={onSelect}
            onNavigate={onNavigate}
          />
        ))}
      </svg>
    </div>
  );
}

function MemoryTimeline({ domain, path, allPaths, stats, onNavigate }) {
  const [items, setItems] = useState([]);
  const [selectedKey, setSelectedKey] = useState('');
  const [expanded, setExpanded] = useState(() => new Set());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const leafPaths = useMemo(() => selectTimelineLeaves(allPaths, domain, path), [allPaths, domain, path]);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError('');

    Promise.all(
      leafPaths.slice(0, 80).map((entry) =>
        getNode(entry.path, entry.domain)
          .then((data) => ({
            ...entry,
            created_at: data.node?.created_at,
            content: data.node?.content || '',
            parsed: parseMemoryContent(data.node?.content || ''),
            priority: data.node?.priority,
            disclosure: data.node?.disclosure,
          }))
          .catch(() => ({
            ...entry,
            created_at: null,
            content: '',
          })),
      ),
    )
      .then((next) => {
        if (!alive) return;
        setItems(next.filter((item) => item.created_at));
      })
      .catch((err) => {
        if (alive) setError(err.response?.data?.detail || err.message || 'Unable to build timeline');
      })
      .finally(() => {
        if (alive) setLoading(false);
      });

    return () => {
      alive = false;
    };
  }, [leafPaths]);

  const groups = useMemo(() => groupTimelineItems(items), [items]);
  const selectedItem = items.find((item) => timelineKey(item) === selectedKey) || items[0];

  useEffect(() => {
    setExpanded(new Set(groups.slice(0, 2).map((group) => group.key)));
  }, [groups]);

  useEffect(() => {
    if (items.length && !items.some((item) => timelineKey(item) === selectedKey)) {
      setSelectedKey(timelineKey(items[0]));
    }
  }, [items, selectedKey]);

  function toggleGroup(key) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  return (
    <section className="surface overflow-hidden rounded-lg">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 p-4">
        <div>
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-950">
            <ListTree size={17} className="text-cyan-300" />
            Memory Timeline
          </div>
          <div className="mt-1 text-xs text-slate-500">
            Leaf memories grouped by day for recent records, and by month for older records.
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <StatusPill tone="cyan">{stats.sessions} sessions</StatusPill>
          <StatusPill tone="green">{stats.episodic} episodic paths</StatusPill>
          <StatusPill>{stats.semantic} semantic paths</StatusPill>
        </div>
      </div>

      <div className="p-4">
        {loading ? (
          <div className="flex min-h-72 items-center justify-center">
            <LoadingState label="Loading memory timeline" />
          </div>
        ) : error ? (
          <EmptyState icon={AlertTriangle} title="Timeline unavailable" detail={error} />
        ) : groups.length ? (
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
            <div className="relative max-h-[720px] overflow-y-auto pr-2 pl-5 before:absolute before:bottom-0 before:left-2 before:top-0 before:w-px before:bg-slate-100">
              <div className="space-y-4">
                {groups.map((group) => {
                  const open = expanded.has(group.key);
                  return (
                    <article key={group.key} className="relative">
                      <span className="absolute -left-[18px] top-4 h-3 w-3 rounded-full border-2 border-white bg-cyan-400" />
                      <button
                        type="button"
                        onClick={() => toggleGroup(group.key)}
                        className="mb-2 flex w-full items-center justify-between gap-3 rounded-md border border-slate-200 bg-white px-3 py-2 text-left hover:border-cyan-800"
                      >
                        <div>
                          <div className="text-sm font-semibold text-slate-950">{group.label}</div>
                          <div className="mt-1 text-xs text-slate-500">{group.mode === 'month' ? 'Monthly archive' : 'Daily memory group'}</div>
                        </div>
                        <StatusPill tone={group.mode === 'month' ? 'amber' : 'cyan'}>{group.items.length} memories</StatusPill>
                      </button>
                      {open ? (
                        <div className="grid gap-2">
                          {group.items.map((item) => (
                            <TimelineCard
                              key={timelineKey(item)}
                              item={item}
                              active={timelineKey(item) === timelineKey(selectedItem)}
                              onSelect={() => setSelectedKey(timelineKey(item))}
                            />
                          ))}
                        </div>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            </div>
            <TimelineDetail item={selectedItem} onNavigate={onNavigate} />
          </div>
        ) : (
          <EmptyState icon={ListTree} title="No timed leaf memories" detail="The current selection has no leaf nodes with created_at exposed by the API." />
        )}
      </div>
    </section>
  );
}

function TimelineCard({ item, active, onSelect }) {
  const type = memoryTypeFromPath(item.path);
  const summary = memorySummary(item);
  return (
    <button
      type="button"
      onClick={onSelect}
      className={clsx(
        'rounded-md border p-3 text-left transition hover:border-cyan-700/60 hover:bg-white',
        active ? 'border-cyan-700/70 bg-cyan-50' : 'border-slate-200 bg-white',
      )}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="truncate text-sm font-semibold text-slate-950">{summary.title}</span>
        <div className="flex shrink-0 gap-1">
          <StatusPill tone={item.domain === 'semantic' ? 'neutral' : 'green'}>{item.domain}</StatusPill>
          <StatusPill tone="amber">{type}</StatusPill>
        </div>
      </div>
      <div className="mb-2 text-xs text-slate-500">{formatExactDate(item.created_at)}</div>
      <div className="flex flex-wrap gap-1">
        {summary.status ? <StatusPill tone={summary.status === 'completed' ? 'green' : 'amber'}>{summary.status}</StatusPill> : null}
        {summary.method ? <StatusPill>{summary.method}</StatusPill> : null}
      </div>
    </button>
  );
}

function TimelineDetail({ item, onNavigate }) {
  if (!item) {
    return (
      <aside className="rounded-lg border border-slate-200 bg-white p-4">
        <EmptyState icon={ListTree} title="Select a memory" />
      </aside>
    );
  }

  const summary = memorySummary(item);
  const type = memoryTypeFromPath(item.path);

  return (
    <aside className="sticky top-4 max-h-[720px] self-start overflow-y-auto rounded-lg border border-slate-200 bg-white p-4">
      <div className="mb-4">
        <div className="text-xs font-semibold uppercase tracking-wide text-cyan-300">Memory Detail</div>
        <h3 className="mt-2 text-lg font-semibold text-slate-950">{summary.title}</h3>
        <div className="mt-2 flex flex-wrap gap-2">
          <StatusPill tone={item.domain === 'semantic' ? 'neutral' : 'green'}>{item.domain}</StatusPill>
          <StatusPill tone="amber">{type}</StatusPill>
          {summary.status ? <StatusPill tone={summary.status === 'completed' ? 'green' : 'amber'}>{summary.status}</StatusPill> : null}
        </div>
      </div>

      <div className="grid gap-2">
        <FieldCard label="Time" value={formatExactDate(item.created_at)} />
        {summary.skill ? <FieldCard label="Skill" value={summary.skill} /> : null}
        {summary.method ? <FieldCard label="Method" value={summary.method} /> : null}
        {item.disclosure ? <FieldCard label="Note" value={item.disclosure} /> : null}
      </div>

      {summary.parameters ? (
        <FieldCard label="Parameters" value={summary.parameters} code tall />
      ) : null}

      <StructuredMemoryContent summary={summary} />

      <button
        type="button"
        onClick={() => onNavigate(item.path, item.domain)}
        className="mt-4 inline-flex h-9 items-center gap-2 rounded-md border border-slate-300 bg-white px-3 text-sm font-medium text-slate-700 hover:border-cyan-700 hover:text-cyan-100"
      >
        <Link2 size={15} />
        Open node
      </button>
    </aside>
  );
}

function StructuredMemoryContent({ summary }) {
  const rows = [
    ['Status', summary.status],
    ['Output Path', summary.output],
    ['Source Dataset', summary.sourceDataset],
    ['Parent Analysis', summary.parentAnalysis],
    ['Duration', summary.duration],
    ['Importance', summary.importance],
    ['Confidence', summary.confidence],
    ['Evidence', summary.evidence],
  ].filter(([, value]) => value !== undefined && value !== null && value !== '');

  if (!rows.length && !summary.note) return null;

  return (
    <div className="mt-2 grid gap-2">
      {rows.map(([label, value]) => (
        <FieldCard key={label} label={label} value={value} code={isLongTechnicalValue(value)} />
      ))}
      {summary.note ? <FieldCard label="Task / Note" value={summary.note} tall /> : null}
      {summary.trajectory ? <FieldCard label="Task Trajectory" value={summary.trajectory} code tall /> : null}
      {summary.keySteps ? <FieldCard label="Key Steps" value={summary.keySteps} code tall /> : null}
    </div>
  );
}

function FieldCard({ label, value, code = false, tall = false }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-100 p-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-cyan-300">
        <span className="text-slate-500">{'{}'}</span>
        {label}
      </div>
      <pre
        className={clsx(
          'whitespace-pre-wrap break-words text-sm leading-6 text-slate-700',
          code && 'overflow-x-auto font-mono text-xs',
          tall ? 'max-h-64 overflow-y-auto' : 'max-h-28 overflow-y-auto',
        )}
      >
        {formatDisplayText(value)}
      </pre>
    </div>
  );
}

function GraphNode({ item, active, onSelect, onNavigate }) {
  const canOpen = item.path !== undefined && item.kind !== 'alias';

  function handleDoubleClick() {
    if (canOpen) onNavigate(item.path, item.domain);
  }

  return (
    <g
      tabIndex="0"
      role="button"
      aria-label={item.label}
      onClick={() => onSelect(item.id)}
      onDoubleClick={handleDoubleClick}
      className="cursor-pointer outline-none"
    >
      <circle
        cx={item.x}
        cy={item.y}
        r={active ? 34 : 28}
        className={clsx('stroke-2 transition', kindClasses[item.kind] || 'fill-slate-500 stroke-slate-200')}
      />
      <circle cx={item.x} cy={item.y} r={active ? 42 : 36} className={clsx('fill-transparent stroke-2', active ? 'stroke-cyan-300/70' : 'stroke-slate-300/70')} />
      <text x={item.x} y={item.y + 52} textAnchor="middle" className="pointer-events-none select-none fill-slate-900 text-[13px] font-semibold">
        {formatNodeLabel(item.label)}
      </text>
      <text x={item.x} y={item.y + 68} textAnchor="middle" className="pointer-events-none select-none fill-slate-500 text-[10px]">
        {item.kind}
      </text>
    </g>
  );
}

function SelectionDetailPanel({ graph, selected, onSelect, onNavigate }) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const outgoing = useMemo(
    () => graph.edges
      .filter((edge) => edge.source === selected?.id)
      .map((edge) => ({
        edge,
        node: graph.nodes.find((item) => item.id === edge.target),
      }))
      .filter((item) => item.node),
    [graph, selected],
  );
  const isLeaf = selected && outgoing.length === 0;
  const canFetchDetail = isLeaf && selected?.domain && selected?.path !== undefined;

  useEffect(() => {
    let alive = true;
    setDetail(null);
    setError('');

    if (!canFetchDetail) {
      setLoading(false);
      return () => {
        alive = false;
      };
    }

    setLoading(true);
    getNode(selected.path || '', selected.domain)
      .then((data) => {
        if (alive) setDetail(data);
      })
      .catch((err) => {
        if (alive) setError(err.response?.data?.detail || err.message || 'Unable to load node detail');
      })
      .finally(() => {
        if (alive) setLoading(false);
      });

    return () => {
      alive = false;
    };
  }, [canFetchDetail, selected?.domain, selected?.path]);

  if (!selected) return null;

  return (
    <div className="border-t border-slate-200 bg-white p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-950">
            <ListTree size={16} className="text-cyan-300" />
            {isLeaf ? 'Node detail' : 'Connected items'}
          </div>
        <div className="mt-1 break-all text-xs text-slate-500">{formatDisplayText(selected.subtitle)}</div>
        </div>
        <StatusPill tone={isLeaf ? 'cyan' : 'green'}>{isLeaf ? 'leaf' : `${outgoing.length} items`}</StatusPill>
      </div>

      {isLeaf ? (
        <LeafDetail
          selected={selected}
          detail={detail}
          loading={loading}
          error={error}
          onNavigate={onNavigate}
        />
      ) : (
        <ConnectedList items={outgoing} onSelect={onSelect} onNavigate={onNavigate} />
      )}
    </div>
  );
}

function LeafDetail({ selected, detail, loading, error, onNavigate }) {
  const node = detail?.node;
  const content = node?.content || selected.content || '';
  const canOpen = selected.path !== undefined && selected.domain;
  const title = node ? formatNodeTitle(node) : selected.label;

  if (loading) {
    return <LoadingState label="Loading selected memory" />;
  }

  return (
    <div className="grid gap-3 lg:grid-cols-[1fr_auto]">
      <div className="min-w-0">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <div className="text-base font-semibold text-slate-950">{title}</div>
          {node?.priority !== undefined || selected.priority !== undefined ? (
            <StatusPill tone="cyan">priority {node?.priority ?? selected.priority}</StatusPill>
          ) : null}
        </div>
        {error ? <div className="mb-2 text-sm text-amber-700">{error}</div> : null}
        {node?.disclosure ? (
          <div className="mb-3 rounded-md border border-amber-700/40 bg-amber-50 p-3 text-sm text-amber-100">
            {formatDisplayText(node.disclosure)}
          </div>
        ) : null}
        {content ? (
          <pre className="mono-box max-h-56 overflow-y-auto whitespace-pre-wrap">{formatDisplayText(content)}</pre>
        ) : (
          <div className="rounded-md border border-slate-200 bg-white p-3 text-sm text-slate-500">
            No content is exposed for this selected node.
          </div>
        )}
      </div>
      {canOpen ? (
        <button
          type="button"
          onClick={() => onNavigate(selected.path, selected.domain)}
          className="inline-flex h-9 items-center gap-2 rounded-md border border-slate-300 bg-white px-3 text-sm font-medium text-slate-700 hover:border-cyan-700 hover:text-cyan-100"
        >
          <Link2 size={15} />
          Open
        </button>
      ) : null}
    </div>
  );
}

function ConnectedList({ items, onSelect, onNavigate }) {
  return (
    <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
      {items.map(({ edge, node }) => (
        <button
          key={edge.id}
          type="button"
          onClick={() => onSelect(node.id)}
          onDoubleClick={() => {
            if (node.path !== undefined && node.domain) onNavigate(node.path, node.domain);
          }}
          className="rounded-md border border-slate-200 bg-white p-3 text-left transition hover:border-cyan-700/60 hover:bg-white"
        >
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="truncate text-sm font-semibold text-slate-950">{node.label}</span>
            <StatusPill>{edge.label}</StatusPill>
          </div>
          <div className="truncate text-xs text-slate-500">{formatDisplayText(node.subtitle)}</div>
        </button>
      ))}
    </div>
  );
}

function NodeInspector({ node, onNavigate }) {
  const canOpen = node.path !== undefined && node.kind !== 'alias';

  return (
    <div className="space-y-3">
      <div>
        <div className="text-base font-semibold text-slate-950">{formatDisplayText(node.label)}</div>
        <div className="mt-1 break-all text-xs text-slate-500">{formatDisplayText(node.subtitle)}</div>
      </div>
      {node.content ? <div className="mono-box max-h-48 overflow-y-auto">{formatDisplayText(node.content)}</div> : null}
      {node.priority !== undefined && node.priority !== null ? <StatusPill tone="cyan">priority {node.priority}</StatusPill> : null}
      {canOpen ? (
        <button
          type="button"
          onClick={() => onNavigate(node.path, node.domain)}
          className="inline-flex h-9 items-center gap-2 rounded-md border border-slate-300 bg-white px-3 text-sm font-medium text-slate-700 hover:border-cyan-700 hover:text-cyan-100"
        >
          <Link2 size={15} />
          Open node
        </button>
      ) : null}
    </div>
  );
}

function Legend({ color, label }) {
  return (
    <div className="flex items-center gap-2">
      <span className={clsx('h-2.5 w-2.5 rounded-full', color)} />
      <span>{label}</span>
    </div>
  );
}

function positionNodes(nodes) {
  const center = { x: 500, y: 300 };
  const groups = {
    current: nodes.filter((item) => item.kind === 'current'),
    child: nodes.filter((item) => item.kind === 'child'),
    alias: nodes.filter((item) => item.kind === 'alias'),
    keyword: nodes.filter((item) => item.kind === 'keyword'),
    sessionHub: nodes.filter((item) => item.kind === 'sessionHub'),
    session: nodes.filter((item) => item.kind === 'session'),
    episodic: nodes.filter((item) => item.kind === 'episodic'),
    semantic: nodes.filter((item) => item.kind === 'semantic'),
    memoryType: nodes.filter((item) => item.kind === 'memoryType'),
    memoryLeaf: nodes.filter((item) => item.kind === 'memoryLeaf'),
  };

  if (groups.sessionHub.length) {
    return [
      ...groups.sessionHub.map((item) => ({ ...item, ...center })),
      ...ring(groups.session, center, 190, 185, 355),
      ...ring(groups.episodic, { x: 360, y: 315 }, 170, 210, 330),
      ...ring(groups.semantic, { x: 640, y: 315 }, 170, 210, 330),
      ...ring(groups.memoryType, center, 255, 15, 165),
      ...ring(groups.memoryLeaf, { x: 500, y: 440 }, 290, 195, 345),
    ];
  }

  return [
    ...groups.current.map((item) => ({ ...item, ...center })),
    ...ring(groups.child, center, 205, -110, 220),
    ...ring(groups.keyword, center, 270, 190, 320),
    ...ring(groups.alias, center, 280, -25, 85),
  ];
}

function formatNodeLabel(label = '') {
  const text = String(label);
  return text.length > 18 ? `${text.slice(0, 17)}...` : text;
}

function parseMemoryContent(content = '') {
  if (!content || typeof content !== 'string') return null;
  try {
    const parsed = JSON.parse(content);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

function memorySummary(item) {
  const data = item.parsed || {};
  const type = memoryTypeFromPath(item.path);
  const skill = data.skill || '';
  const status = data.status || '';
  const method = data.method || '';
  const output = data.output_path || data.file_path || '';
  const leafName = formatNodeTitle(item) || item.path.split('/').filter(Boolean).pop() || type;
  const title = skill || data.key || data.project_goal || data.biological_label || readableLeafName(leafName);
  const parameters = data.parameters ? JSON.stringify(data.parameters, null, 2) : '';
  const duration = data.duration_seconds !== undefined ? `${Number(data.duration_seconds).toFixed(1)}s` : '';
  const importance = data.importance !== undefined ? String(data.importance) : '';
  const confidence = data.confidence_score !== undefined ? String(data.confidence_score) : data.confidence || '';
  const note = data.fail_reason || data.task_main || data.evidence || '';

  return {
    title,
    skill,
    status,
    method,
    output,
    parameters,
    sourceDataset: data.source_dataset_id,
    parentAnalysis: data.parent_analysis_id,
    duration,
    importance,
    confidence,
    evidence: data.evidence_uri || data.source_analysis_id,
    note,
    trajectory: data.task_trajectory,
    keySteps: data.key_steps,
    content: item.content || item.uri || formatMemoryUri(item.domain, item.path),
  };
}

function isLongTechnicalValue(value) {
  const text = String(value || '');
  return text.length > 36 || text.includes('/') || text.includes('\\') || /^[a-f0-9]{16,}$/i.test(text);
}

function timelineKey(item) {
  return `${item?.domain || ''}:${item?.path || ''}`;
}

function readableLeafName(value = '') {
  const text = String(value);
  if (/^[a-f0-9]{16,}$/i.test(text)) return 'Memory item';
  return text.replace(/[-_]/g, ' ');
}

function selectTimelineLeaves(allPaths, domain, path) {
  const layerDomains = new Set(['session', 'episodic', 'semantic']);
  const scoped = allPaths.filter((entry) => layerDomains.has(entry.domain) && entry.path);
  const pathSet = new Set(scoped.map((entry) => `${entry.domain}:${entry.path}`));
  const leaves = scoped.filter((entry) => {
    const prefix = `${entry.domain}:${entry.path}/`;
    return !Array.from(pathSet).some((value) => value.startsWith(prefix));
  });

  if (!path) return leaves;

  const parts = path.split('/').filter(Boolean);
  const sessionId = parts[0];
  const type = parts[1] || '';
  const branch = parts.length > 2 ? path : '';
  const selectedHasChildren = scoped.some((entry) =>
    entry.domain === domain && entry.path?.startsWith(`${path}/`),
  );

  return leaves.filter((entry) => {
    if (!entry.path.startsWith(`${sessionId}/`) && entry.path !== sessionId) return false;
    if (type && memoryTypeFromPath(entry.path) !== type) return false;
    if (branch && selectedHasChildren) return entry.path.startsWith(`${path}/`);
    if (branch && entry.domain === domain) return entry.path === path;
    if (branch) {
      return pathTailAfterType(entry.path, sessionId, type) === pathTailAfterType(path, sessionId, type);
    }
    return true;
  });
}

function groupTimelineItems(items) {
  const now = Date.now();
  const monthMs = 30 * 24 * 60 * 60 * 1000;
  const groups = new Map();

  items
    .slice()
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
    .forEach((item) => {
      const date = new Date(item.created_at);
      const old = now - date.getTime() > monthMs;
      const key = old ? `month:${date.getFullYear()}-${pad(date.getMonth() + 1)}` : `day:${toDateKey(date)}`;
      const label = old ? `${date.getFullYear()}-${pad(date.getMonth() + 1)}` : toDateKey(date);
      if (!groups.has(key)) groups.set(key, { key, label, mode: old ? 'month' : 'day', items: [] });
      groups.get(key).items.push(item);
    });

  return Array.from(groups.values());
}

function memoryTypeFromPath(path = '') {
  return path.split('/').filter(Boolean)[1] || 'session';
}

function pathTailAfterType(path, sessionId, type) {
  const prefix = `${sessionId}/${type}/`;
  return path.startsWith(prefix) ? path.slice(prefix.length) : path;
}

function toDateKey(date) {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function formatExactDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'unknown date';
  return `${toDateKey(date)} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function pad(value) {
  return String(value).padStart(2, '0');
}

function ring(items, center, radius, startDeg, endDeg) {
  if (!items.length) return [];
  const span = endDeg - startDeg;
  return items.map((item, index) => {
    const ratio = items.length === 1 ? 0.5 : index / (items.length - 1);
    const angle = ((startDeg + span * ratio) * Math.PI) / 180;
    return {
      ...item,
      x: center.x + Math.cos(angle) * radius,
      y: center.y + Math.sin(angle) * radius,
    };
  });
}
