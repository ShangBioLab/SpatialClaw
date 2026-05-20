import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronRight, Clock3, Database, FileText, Layers, Server } from 'lucide-react';
import clsx from 'clsx';

import { api } from '../../../lib/memoryApi';
import { formatDisplayText, formatLeafLabel, formatSessionId } from '../../../shared/utils/content';
import { layerTypeChildren, shouldAutoRevealForRouteChange, visibleChildCount } from '../utils/sessionNavigation';

function isSessionActive(activeDomain, activePath, sessionId) {
  return activeDomain === 'session' && activePath === sessionId;
}

function isSessionAncestor(activeDomain, activePath, sessionId) {
  if (activeDomain === 'session') return activePath === sessionId;
  return ['episodic', 'semantic'].includes(activeDomain) && (
    activePath === sessionId || activePath.startsWith(`${sessionId}/`)
  );
}

function isLayerActive(activeDomain, activePath, sessionId, layer) {
  return activeDomain === layer && activePath === sessionId;
}

function isLayerAncestor(activeDomain, activePath, sessionId, layer) {
  return activeDomain === layer && (
    activePath === sessionId || activePath.startsWith(`${sessionId}/`)
  );
}

function layerLabel(layer) {
  return layer === 'episodic' ? 'Episodic' : 'Semantic';
}

function layerIcon(layer) {
  return layer === 'episodic' ? Clock3 : Database;
}

function TreeRow({
  active,
  muted = false,
  icon: Icon,
  label,
  count,
  level,
  canExpand,
  expanded,
  loading,
  onClick,
  onToggle,
}) {
  const badgeCount = visibleChildCount(count);

  return (
    <div
      className={clsx(
        'flex cursor-pointer items-center gap-1.5 rounded-xl border border-transparent py-2 pr-2 text-sm transition-all duration-300 group',
        active ? 'bg-indigo-50 text-indigo-700 shadow-sm border-indigo-200' : 'text-slate-500 hover:bg-slate-50 hover:text-slate-800 hover:border-slate-200',
        muted && !active && 'opacity-75',
      )}
      style={{ paddingLeft: `${level * 12 + 8}px` }}
      onClick={onClick}
    >
      <button
        type="button"
        className="flex h-5 w-5 shrink-0 items-center justify-center"
        onClick={(event) => {
          event.stopPropagation();
          if (canExpand) onToggle?.();
        }}
      >
        {loading ? (
          <span className="h-3 w-3 rounded-full border-2 border-slate-500 border-t-transparent animate-spin" />
        ) : canExpand ? (
          <ChevronRight size={14} className={clsx('transition-transform text-slate-500 group-hover:text-slate-700', expanded && 'rotate-90')} />
        ) : null}
      </button>
      <Icon size={14} className={clsx('shrink-0', active ? 'text-indigo-600' : 'text-slate-400 group-hover:text-slate-600')} />
      <span className="min-w-0 flex-1 truncate text-[13px]">{label}</span>
      {badgeCount !== undefined ? (
        <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-600">{badgeCount}</span>
      ) : null}
    </div>
  );
}

function MemoryBranchNode({ domain, path, name, childrenCount, activeDomain, activePath, onNavigate, level }) {
  const isActive = activeDomain === domain && activePath === path;
  const isAncestor = activeDomain === domain && activePath.startsWith(`${path}/`);
  const [expanded, setExpanded] = useState(isAncestor || isActive);
  const [children, setChildren] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fetched, setFetched] = useState(false);
  const prevActive = useRef({ domain: activeDomain, path: activePath });
  const hasChildren = fetched ? children.length > 0 : (childrenCount === undefined || childrenCount > 0);

  useEffect(() => {
    if (expanded && !fetched && hasChildren) {
      setLoading(true);
      api.get('/browse/node', { params: { domain, path, nav_only: true } })
        .then((res) => {
          setChildren(res.data.children || []);
          setFetched(true);
        })
        .catch(() => {
          setChildren([]);
          setFetched(true);
        })
        .finally(() => setLoading(false));
    }
  }, [domain, expanded, fetched, hasChildren, path]);

  useEffect(() => {
    if (
      shouldAutoRevealForRouteChange(
        prevActive.current,
        { domain: activeDomain, path: activePath },
        isActive || isAncestor,
        expanded,
      )
    ) {
      setExpanded(true);
    }
    prevActive.current = { domain: activeDomain, path: activePath };
  }, [activeDomain, activePath, expanded, isActive, isAncestor]);

  return (
    <div>
      <TreeRow
        active={isActive}
        icon={FileText}
        label={formatLeafLabel(path, name)}
        count={childrenCount}
        level={level}
        canExpand={hasChildren}
        expanded={expanded}
        loading={loading}
        onToggle={() => setExpanded((value) => !value)}
        onClick={() => {
          onNavigate(path, domain);
          if (!expanded && hasChildren) setExpanded(true);
        }}
      />
      {expanded && children.length ? (
        <div>
          {children.map((child) => (
            <MemoryBranchNode
              key={`${child.domain || domain}:${child.path}`}
              domain={child.domain || domain}
              path={child.path}
              name={child.name}
              childrenCount={child.approx_children_count}
              activeDomain={activeDomain}
              activePath={activePath}
              onNavigate={onNavigate}
              level={level + 1}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function LayerNode({ sessionId, layer, activeDomain, activePath, onNavigate, level }) {
  const Icon = layerIcon(layer.domain);
  const active = isLayerActive(activeDomain, activePath, sessionId, layer.domain);
  const ancestor = isLayerAncestor(activeDomain, activePath, sessionId, layer.domain);
  const [expanded, setExpanded] = useState(ancestor || active);
  const prevActive = useRef({ domain: activeDomain, path: activePath });
  const virtualChildren = useMemo(() => layerTypeChildren(layer), [layer]);
  const count = layer.pathCount;
  const hasChildren = virtualChildren.length > 0;

  useEffect(() => {
    if (
      shouldAutoRevealForRouteChange(
        prevActive.current,
        { domain: activeDomain, path: activePath },
        active || ancestor,
        expanded,
      )
    ) {
      setExpanded(true);
    }
    prevActive.current = { domain: activeDomain, path: activePath };
  }, [active, activeDomain, activePath, ancestor, expanded]);

  return (
    <div>
      <TreeRow
        active={active}
        muted={!count}
        icon={Icon}
        label={layerLabel(layer.domain)}
        count={count}
        level={level}
        canExpand={hasChildren}
        expanded={expanded}
        onToggle={() => setExpanded((value) => !value)}
        onClick={() => {
          if (expanded && (active || ancestor)) {
            setExpanded(false);
            return;
          }
          if (layer.exists) onNavigate(sessionId, layer.domain);
          if (!expanded && hasChildren) setExpanded(true);
        }}
      />
      {expanded && hasChildren ? (
        <div>
          {virtualChildren.map((child) => (
            <MemoryBranchNode
              key={`${child.domain}:${child.path}`}
              domain={child.domain}
              path={child.path}
              name={child.name}
              childrenCount={child.approx_children_count}
              activeDomain={activeDomain}
              activePath={activePath}
              onNavigate={onNavigate}
              level={level + 1}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function SessionNode({ session, activeDomain, activePath, onNavigate }) {
  const active = isSessionActive(activeDomain, activePath, session.id);
  const ancestor = isSessionAncestor(activeDomain, activePath, session.id);
  const [expanded, setExpanded] = useState(active || ancestor);
  const prevActive = useRef({ domain: activeDomain, path: activePath });

  useEffect(() => {
    if (
      shouldAutoRevealForRouteChange(
        prevActive.current,
        { domain: activeDomain, path: activePath },
        active || ancestor,
        expanded,
      )
    ) {
      setExpanded(true);
    }
    prevActive.current = { domain: activeDomain, path: activePath };
  }, [active, activeDomain, activePath, ancestor, expanded]);

  return (
    <div>
      <TreeRow
        active={active}
        icon={Server}
        label={formatSessionId(session.id)}
        count={session.totalPathCount}
        level={0}
        canExpand
        expanded={expanded}
        onToggle={() => setExpanded((value) => !value)}
        onClick={() => {
          if (expanded && (active || ancestor)) {
            setExpanded(false);
            return;
          }
          if (session.sessionNode) onNavigate(session.id, 'session');
          if (!expanded) setExpanded(true);
        }}
      />
      {expanded ? (
        <div>
          <LayerNode
            sessionId={session.id}
            layer={session.layers.episodic}
            activeDomain={activeDomain}
            activePath={activePath}
            onNavigate={onNavigate}
            level={1}
          />
          <LayerNode
            sessionId={session.id}
            layer={session.layers.semantic}
            activeDomain={activeDomain}
            activePath={activePath}
            onNavigate={onNavigate}
            level={1}
          />
        </div>
      ) : null}
    </div>
  );
}

export default function SessionMemoryTree({ sessions, activeDomain, activePath, onNavigate, loading = false }) {
  return (
    <section>
      <div className="mb-2 px-1">
        <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
          <Layers size={12} />
          Sessions
        </div>
        <div className="text-[11px] text-slate-400">Open a session, then inspect episodic and semantic memory.</div>
      </div>
      {loading ? (
        <div className="rounded-md border border-slate-200 bg-white p-3 text-xs text-slate-500">
          Loading sessions
        </div>
      ) : sessions.length ? (
        <div className="space-y-2">
          {sessions.map((session) => (
            <SessionNode
              key={session.id}
              session={session}
              activeDomain={activeDomain}
              activePath={activePath}
              onNavigate={onNavigate}
            />
          ))}
        </div>
      ) : (
        <div className="rounded-md border border-slate-200 bg-white p-3 text-xs text-slate-500">
          {formatDisplayText('No memory sessions yet')}
        </div>
      )}
    </section>
  );
}
