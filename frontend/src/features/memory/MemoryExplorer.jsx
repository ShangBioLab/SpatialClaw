import { useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  AlertTriangle,
  Braces,
  Edit3,
  FileText,
  FolderTree,
  Network,
  RefreshCw,
  Save,
  X,
} from 'lucide-react';

import { getAllPaths, getDomains, getNode, updateNode } from '../../lib/memoryApi';
import Button from '../../shared/components/Button';
import EmptyState from '../../shared/components/EmptyState';
import IconButton from '../../shared/components/IconButton';
import LoadingState from '../../shared/components/LoadingState';
import PageHeader from '../../shared/components/PageHeader';
import StatusPill from '../../shared/components/StatusPill';
import {
  formatDisplayText,
  formatMemoryUri,
  formatNodeTitle,
  formatPathLabel,
  parseJsonObject,
  shortPath,
} from '../../shared/utils/content';
import MemoryBreadcrumb from './components/MemoryBreadcrumb';
import TermMarker from './components/TermMarker';
import TermEditor from './components/TermEditor';
import MemoryTree from './components/MemoryTree';
import MemoryNodeCard from './components/MemoryNodeCard';
import RankIndicator from './components/RankIndicator';
import SessionMemoryTree from './components/SessionMemoryTree';
import GraphView from './graph/GraphView';
import { buildSessionNavigation, preferredLayeredViewMode } from './utils/sessionNavigation';

const LAYERED_DOMAINS = new Set(['session', 'episodic', 'semantic']);

export default function MemoryExplorer() {
  const [searchParams, setSearchParams] = useSearchParams();
  const domain = searchParams.get('domain') || 'session';
  const path = searchParams.get('path') || '';

  const [domains, setDomains] = useState([]);
  const [allPaths, setAllPaths] = useState([]);
  const [navLoading, setNavLoading] = useState(true);
  const [data, setData] = useState({ node: null, children: [], breadcrumbs: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [editing, setEditing] = useState(false);
  const [viewMode, setViewMode] = useState('detail');
  const [draft, setDraft] = useState({ content: '', disclosure: '', priority: 0 });
  const [saving, setSaving] = useState(false);
  const routeRef = useRef({ domain, path });

  useEffect(() => {
    routeRef.current = { domain, path };
  }, [domain, path]);

  useEffect(() => {
    let alive = true;
    setNavLoading(true);
    Promise.all([
      getDomains().catch(() => []),
      getAllPaths().then((data) => data.paths || []).catch(() => []),
    ])
      .then(([nextDomains, nextPaths]) => {
        if (!alive) return;
        setDomains(nextDomains);
        setAllPaths(nextPaths);
      })
      .finally(() => {
        if (alive) setNavLoading(false);
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    loadNode();
  }, [domain, path]);

  useEffect(() => {
    const nextViewMode = preferredLayeredViewMode({
      domain,
      path,
      node: data.node,
      children: data.children,
      editing,
    });

    if (nextViewMode) {
      setViewMode(nextViewMode);
    }
  }, [domain, path, data.node, data.children, editing]);

  async function loadNode() {
    setLoading(true);
    setError('');
    setEditing(false);
    try {
      const next = await getNode(path, domain);
      if (routeRef.current.domain === domain && routeRef.current.path === path) {
        setData(next);
        setDraft({
          content: next.node?.content || '',
          disclosure: next.node?.disclosure || '',
          priority: next.node?.priority ?? 0,
        });
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Unable to load node');
    } finally {
      setLoading(false);
    }
  }

  function navigateTo(nextPath, nextDomain = domain) {
    const params = new URLSearchParams();
    params.set('domain', nextDomain);
    if (nextPath) params.set('path', nextPath);
    setSearchParams(params);
  }

  function startEditing() {
    setDraft({
      content: data.node?.content || '',
      disclosure: data.node?.disclosure || '',
      priority: data.node?.priority ?? 0,
    });
    setEditing(true);
    setViewMode('detail');
  }

  async function saveDraft() {
    const node = data.node;
    const payload = {};
    if (draft.content !== (node?.content || '')) payload.content = draft.content;
    if (draft.disclosure !== (node?.disclosure || '')) payload.disclosure = draft.disclosure;
    if (draft.priority !== (node?.priority ?? 0)) payload.priority = draft.priority;
    if (!Object.keys(payload).length) {
      setEditing(false);
      return;
    }

    setSaving(true);
    try {
      await updateNode(path, domain, payload);
      await loadNode();
      setEditing(false);
    } catch (err) {
      window.alert(`Save failed: ${err.message}`);
    } finally {
      setSaving(false);
    }
  }

  const node = data.node;
  const isRoot = !path;
  const jsonContent = useMemo(() => parseJsonObject(node?.content), [node?.content]);
  const sessionNavigation = useMemo(() => buildSessionNavigation(allPaths), [allPaths]);

  return (
    <div className="flex h-full overflow-hidden bg-white">
      <aside className="hidden w-80 shrink-0 border-r border-slate-200 bg-white p-4 md:flex md:flex-col">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <div className="text-sm font-semibold text-slate-950">Knowledge Graph</div>
            <div className="text-xs text-slate-500">{formatMemoryUri(domain, path)}</div>
          </div>
          <IconButton icon={RefreshCw} label="Refresh" onClick={loadNode} />
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto pr-1">
          <SessionMemoryTree
            sessions={sessionNavigation}
            activeDomain={domain}
            activePath={path}
            onNavigate={navigateTo}
            loading={navLoading}
          />
          <OtherDomainTrees
            domains={domains}
            activeDomain={domain}
            activePath={path}
            onNavigate={navigateTo}
          />
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col">
        <PageHeader
          eyebrow="Memory Explorer"
          title={formatNodeTitle(node) || formatPathLabel(path)}
          meta={<MemoryBreadcrumb items={data.breadcrumbs || []} onNavigate={navigateTo} />}
          actions={
            <>
              <StatusPill tone="cyan">{domain}</StatusPill>
              <div className="inline-flex h-9 rounded-md border border-slate-200 bg-white p-0.5">
                <button
                  type="button"
                  onClick={() => setViewMode('detail')}
                  className={`rounded px-3 text-sm font-medium transition ${viewMode === 'detail' ? 'bg-slate-100 text-slate-950' : 'text-slate-500 hover:text-slate-700'}`}
                >
                  Detail
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setEditing(false);
                    setViewMode('graph');
                  }}
                  className={`inline-flex items-center gap-1.5 rounded px-3 text-sm font-medium transition ${viewMode === 'graph' ? 'bg-cyan-600 text-white' : 'text-slate-500 hover:text-slate-700'}`}
                >
                  <Network size={15} />
                  {LAYERED_DOMAINS.has(domain) ? 'Timeline' : 'Graph'}
                </button>
              </div>
              <IconButton icon={RefreshCw} label="Refresh" onClick={loadNode} />
              {editing ? (
                <>
                  <IconButton icon={X} label="Cancel" onClick={() => setEditing(false)} />
                  <Button icon={Save} variant="primary" disabled={saving} onClick={saveDraft}>
                    {saving ? 'Saving' : 'Save'}
                  </Button>
                </>
              ) : node && !node.is_virtual ? (
                <Button icon={Edit3} onClick={startEditing}>Edit</Button>
              ) : null}
            </>
          }
        />

        <div className="min-h-0 flex-1 overflow-y-auto p-5">
          {loading ? (
            <div className="flex h-full items-center justify-center">
              <LoadingState label="Loading memory node" />
            </div>
          ) : error ? (
            <EmptyState
              icon={AlertTriangle}
              title="Node unavailable"
              detail={error}
              action={<Button onClick={() => navigateTo('', domain)}>Open domain root</Button>}
            />
          ) : (
            <div className="mx-auto max-w-7xl space-y-5">
              {viewMode === 'graph' ? (
                <GraphView
                  node={node}
                  children={data.children || []}
                  domain={domain}
                  path={path}
                  onNavigate={navigateTo}
                />
              ) : (
                <>
              {LAYERED_DOMAINS.has(domain) ? (
                <LayeredMemoryExplanation activeDomain={domain} />
              ) : null}

              {node && (!isRoot || !node.is_virtual || editing) ? (
                <article className="surface rounded-lg">
                  <div className="flex flex-col gap-4 border-b border-slate-200 p-4 lg:flex-row lg:items-start lg:justify-between">
                    <div className="min-w-0">
                      <div className="mb-2 flex flex-wrap items-center gap-2">
                        <h2 className="truncate text-base font-semibold text-slate-950">
                          {formatNodeTitle(node) || formatPathLabel(path)}
                        </h2>
                        <RankIndicator priority={editing ? draft.priority : node.priority} />
                      </div>
                      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                        <StatusPill>{formatMemoryUri(domain, path)}</StatusPill>
                        {node.aliases?.length ? <StatusPill>{node.aliases.length} aliases</StatusPill> : null}
                      </div>
                    </div>

                    {!editing && !node.is_virtual ? (
                      <TermEditor
                        keywords={node.glossary_keywords || []}
                        nodeUuid={node.node_uuid}
                        onUpdate={loadNode}
                      />
                    ) : null}
                  </div>

                  {editing ? (
                    <Editor draft={draft} setDraft={setDraft} />
                  ) : (
                    <div className="p-4">
                      {node.disclosure ? (
                        <div className="mb-4 flex items-start gap-2 rounded-md border border-amber-700/40 bg-amber-50 p-3 text-sm text-amber-100">
                          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                          <span>{formatDisplayText(node.disclosure)}</span>
                        </div>
                      ) : null}

                      {jsonContent ? (
                        <JsonGrid value={jsonContent} />
                      ) : (
                        <div className="mono-box whitespace-pre-wrap">
                          <TermMarker
                            content={formatDisplayText(node.content || '')}
                            glossary={node.glossary_matches || []}
                            currentNodeUuid={node.node_uuid}
                            onNavigate={navigateTo}
                          />
                        </div>
                      )}
                    </div>
                  )}
                </article>
              ) : null}

              <section>
                <div className="mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
                    <FolderTree size={16} className="text-cyan-300" />
                    {isRoot ? 'Root nodes' : 'Child nodes'}
                  </div>
                  <StatusPill>{data.children?.length || 0}</StatusPill>
                </div>

                {data.children?.length ? (
                  <div className="data-grid">
                    {data.children.map((child) => (
                      <MemoryNodeCard
                        key={`${child.domain || domain}:${child.path}`}
                        node={child}
                        currentDomain={domain}
                        onClick={() => navigateTo(child.path, child.domain || domain)}
                      />
                    ))}
                  </div>
                ) : (
                  <EmptyState icon={FileText} title="No child nodes" />
                )}
              </section>
                </>
              )}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function LayeredMemoryExplanation({ activeDomain }) {
  const rows = [
    {
      domain: 'session',
      title: 'Session',
      detail: 'Conversation identity and entry point.',
      example: 'session://<session_id>',
    },
    {
      domain: 'episodic',
      title: 'Episodic',
      detail: 'Workflow events saved under the same session id.',
      example: 'episodic://<session_id>/<type>/...',
    },
    {
      domain: 'semantic',
      title: 'Semantic',
      detail: 'Promoted knowledge derived from stable or important memories.',
      example: 'semantic://<session_id>/<type>/...',
    },
  ];

  return (
    <section className="surface rounded-lg p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-slate-950">Layered memory relationship</div>
          <div className="mt-1 text-xs text-slate-500">These three domains are linked by the same session id. Use Graph for the joined view.</div>
        </div>
        <StatusPill tone="cyan">{activeDomain}</StatusPill>
      </div>
      <div className="grid gap-3 lg:grid-cols-3">
        {rows.map((row) => (
          <div
            key={row.domain}
            className={`rounded-md border p-3 ${row.domain === activeDomain ? 'border-cyan-700/60 bg-cyan-50' : 'border-slate-200 bg-white'}`}
          >
            <div className="text-sm font-semibold text-slate-950">{row.title}</div>
            <div className="mt-1 text-xs leading-5 text-slate-500">{row.detail}</div>
            <code className="mt-3 block truncate rounded bg-slate-100 px-2 py-1.5 font-mono text-[11px] text-slate-500">{row.example}</code>
          </div>
        ))}
      </div>
    </section>
  );
}

function OtherDomainTrees({ domains, activeDomain, activePath, onNavigate }) {
  const otherDomains = domains.filter((entry) => !LAYERED_DOMAINS.has(entry.domain) && (entry.root_count > 0 || entry.domain === activeDomain));
  if (!otherDomains.length) return null;

  return (
    <section className="mt-5">
      <div className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500">Other</div>
      {otherDomains.map((entry) => (
        <MemoryTree
          key={entry.domain}
          domain={entry.domain}
          rootCount={entry.root_count}
          activeDomain={activeDomain}
          activePath={activePath}
          onNavigate={onNavigate}
        />
      ))}
    </section>
  );
}

function Editor({ draft, setDraft }) {
  return (
    <div className="grid gap-4 p-4">
      <div className="grid gap-4 md:grid-cols-[160px_1fr]">
        <label className="block">
          <span className="mb-2 block text-xs font-medium uppercase tracking-wide text-slate-500">Priority</span>
          <input
            type="number"
            min="0"
            value={draft.priority}
            onChange={(event) => setDraft((prev) => ({ ...prev, priority: Number.parseInt(event.target.value, 10) || 0 }))}
            className="h-10 w-full rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-950 outline-none focus:border-cyan-500"
          />
        </label>
        <label className="block">
          <span className="mb-2 block text-xs font-medium uppercase tracking-wide text-slate-500">Disclosure</span>
          <input
            value={draft.disclosure}
            onChange={(event) => setDraft((prev) => ({ ...prev, disclosure: event.target.value }))}
            className="h-10 w-full rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-950 outline-none focus:border-cyan-500"
          />
        </label>
      </div>

      <label className="block">
        <span className="mb-2 block text-xs font-medium uppercase tracking-wide text-slate-500">Content</span>
        <textarea
          value={draft.content}
          onChange={(event) => setDraft((prev) => ({ ...prev, content: event.target.value }))}
          spellCheck={false}
          className="h-[420px] w-full resize-y rounded-md border border-slate-300 bg-white p-4 font-mono text-sm leading-6 text-slate-950 outline-none focus:border-cyan-500"
        />
      </label>
    </div>
  );
}

function JsonGrid({ value }) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {Object.entries(value).map(([key, item]) => (
        <div key={key} className="rounded-md border border-slate-200 bg-white p-3">
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-cyan-300/80">
            <Braces size={13} />
            {key.replace(/_/g, ' ')}
          </div>
          <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-6 text-slate-700">
            {formatDisplayText(item)}
          </pre>
        </div>
      ))}
    </div>
  );
}
