import {
  formatDisplayText,
  formatMemoryUri,
  formatNodeTitle,
  formatPathLabel,
  formatSessionId,
  shortPath,
} from '../../../shared/utils/content';

const NODE_LIMIT = 42;
const LAYER_DOMAINS = new Set(['session', 'episodic', 'semantic']);

function makeUri(domain, path) {
  return formatMemoryUri(domain || 'session', path || 'root');
}

function clip(value = '', max = 64) {
  const text = formatDisplayText(value).trim();
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

function pushNode(nodes, node) {
  if (!node?.id || nodes.has(node.id)) return;
  nodes.set(node.id, node);
}

function pushEdge(edges, edge) {
  if (!edge?.source || !edge?.target || edge.source === edge.target) return;
  const id = edge.id || `${edge.source}->${edge.target}:${edge.kind || 'link'}`;
  if (edges.some((item) => item.id === id)) return;
  edges.push({ ...edge, id });
}

function layerTitle(layer) {
  return layer === 'episodic' ? 'Episodic memory' : 'Semantic memory';
}

function pathName(path) {
  return formatPathLabel(path).replace(/_/g, ' ');
}

export function buildMemoryGraph({
  node,
  children = [],
  domain,
  path,
  allPaths = [],
  glossary = [],
}) {
  const nodes = new Map();
  const edges = [];
  const currentId = node?.node_uuid || `virtual:${makeUri(domain, path)}`;
  const currentUri = makeUri(domain, path);
  const visibleNodeIds = new Set([currentId]);
  const visibleUris = new Set([currentUri]);

  pushNode(nodes, {
    id: currentId,
    label: formatNodeTitle({ ...node, path }) || pathName(path),
    subtitle: currentUri,
    kind: 'current',
    content: clip(node?.content, 110),
    priority: node?.priority,
    path,
    domain,
  });

  children.slice(0, 18).forEach((child) => {
    const childDomain = child.domain || domain;
    const childId = child.node_uuid || `path:${makeUri(childDomain, child.path)}`;
    const childUri = makeUri(childDomain, child.path);
    visibleNodeIds.add(childId);
    visibleUris.add(childUri);
    pushNode(nodes, {
      id: childId,
      label: formatNodeTitle(child) || pathName(child.path),
      subtitle: childUri,
      kind: 'child',
      content: clip(child.content_snippet, 90),
      priority: child.priority,
      path: child.path,
      domain: childDomain,
    });
    pushEdge(edges, {
      source: currentId,
      target: childId,
      kind: 'child',
      label: formatNodeTitle(child) || 'child',
    });
  });

  allPaths
    .filter((entry) => entry.node_uuid === node?.node_uuid)
    .filter((entry) => makeUri(entry.domain || domain, entry.path) !== currentUri)
    .slice(0, 8)
    .forEach((entry) => {
      const uri = makeUri(entry.domain || domain, entry.path);
      const aliasId = `alias:${uri}`;
      visibleUris.add(uri);
      pushNode(nodes, {
        id: aliasId,
        label: formatNodeTitle(entry) || pathName(entry.path),
        subtitle: uri,
        kind: 'alias',
        path: entry.path,
        domain: entry.domain || domain,
      });
      pushEdge(edges, {
        source: currentId,
        target: aliasId,
        kind: 'alias',
        label: 'alias',
      });
    });

  glossary.slice(0, 240).forEach((entry) => {
    const linked = (entry.nodes || []).filter((target) => visibleNodeIds.has(target.node_uuid));
    if (!linked.length) return;
    const keywordId = `keyword:${entry.keyword}`;
    pushNode(nodes, {
      id: keywordId,
      label: entry.keyword,
      subtitle: 'glossary keyword',
      kind: 'keyword',
    });
    linked.slice(0, 6).forEach((target) => {
      pushEdge(edges, {
        source: keywordId,
        target: target.node_uuid,
        kind: 'glossary',
        label: 'mentions',
      });
    });
  });

  return {
    nodes: Array.from(nodes.values()).slice(0, NODE_LIMIT),
    edges: edges.filter((edge) => nodes.has(edge.source) && nodes.has(edge.target)),
    stats: {
      children: children.length,
      aliases: allPaths.filter((entry) => entry.node_uuid === node?.node_uuid).length,
      glossary: glossary.filter((entry) =>
        (entry.nodes || []).some((target) => visibleNodeIds.has(target.node_uuid)),
      ).length,
    },
  };
}

export function shouldUseLayeredGraph(domain) {
  return LAYER_DOMAINS.has(domain);
}

export function buildLayeredMemoryGraph({ domain, path, allPaths = [] }) {
  const nodes = new Map();
  const edges = [];
  const activeParts = path.split('/').filter(Boolean);
  const activeType = activeParts[1] || '';
  const activePath = path || '';
  const sessionIds = resolveSessionIds({ domain, path, allPaths });
  const focusedSessionId = sessionIds.length === 1 ? sessionIds[0] : '';
  const rootId = focusedSessionId ? `session-root:${focusedSessionId}` : 'session-root:all';
  const scopedPaths = focusedSessionId
    ? allPaths.filter((entry) => entry.path === focusedSessionId || entry.path?.startsWith(`${focusedSessionId}/`))
    : allPaths;

  pushNode(nodes, {
    id: rootId,
    label: focusedSessionId ? formatSessionId(focusedSessionId) : 'All sessions',
    subtitle: focusedSessionId ? formatMemoryUri('session', focusedSessionId) : 'session memory index',
    kind: 'sessionHub',
    path: focusedSessionId,
    domain: focusedSessionId ? 'session' : undefined,
  });

  sessionIds.slice(0, 8).forEach((sessionId) => {
    const sessionNodeId = focusedSessionId ? rootId : `session:${sessionId}`;
    if (!focusedSessionId) {
      pushNode(nodes, {
        id: sessionNodeId,
        label: formatSessionId(sessionId),
        subtitle: formatMemoryUri('session', sessionId),
        kind: 'session',
        path: sessionId,
        domain: 'session',
      });
      pushEdge(edges, {
        source: rootId,
        target: sessionNodeId,
        kind: 'session',
        label: 'session',
      });
    }

    ['episodic', 'semantic'].forEach((layer) => {
      const layerPaths = getLayerPaths(allPaths, layer, sessionId);
      const layerId = `${layer}:${sessionId}`;
      pushNode(nodes, {
        id: layerId,
        label: layerTitle(layer),
        subtitle: `${layerPaths.length} paths under ${formatMemoryUri(layer, sessionId)}`,
        kind: layer,
        path: sessionId,
        domain: layer,
      });
      pushEdge(edges, {
        source: sessionNodeId,
        target: layerId,
        kind: layer,
        label: layer,
      });

      visibleGroups(groupByType(layerPaths, sessionId), activeType).forEach((group) => {
        const typeId = `${layer}:${sessionId}:${group.type}`;
        pushNode(nodes, {
          id: typeId,
          label: group.type.replace(/_/g, ' '),
          subtitle: `${group.count} records`,
          kind: 'memoryType',
        });
        pushEdge(edges, {
          source: layerId,
          target: typeId,
          kind: 'memoryType',
          label: String(group.count),
        });

        visibleItems(group.items, { activeDomain: domain, activePath, sessionId, activeType, layer }).forEach((entry) => {
          const leafId = `${layer}:${entry.path}`;
          pushNode(nodes, {
            id: leafId,
            label: formatNodeTitle(entry) || pathName(entry.path),
            subtitle: formatMemoryUri(entry.domain, entry.path),
            kind: 'memoryLeaf',
            path: entry.path,
            domain: entry.domain,
          });
          pushEdge(edges, {
            source: typeId,
            target: leafId,
            kind: 'memoryLeaf',
            label: 'item',
          });
        });
      });
    });
  });

  return {
    nodes: Array.from(nodes.values()).slice(0, NODE_LIMIT),
    edges: edges.filter((edge) => nodes.has(edge.source) && nodes.has(edge.target)),
    stats: {
      sessions: sessionIds.length,
      episodic: scopedPaths.filter((entry) => entry.domain === 'episodic').length,
      semantic: scopedPaths.filter((entry) => entry.domain === 'semantic').length,
    },
    layered: true,
  };
}

function resolveSessionIds({ domain, path, allPaths }) {
  if (path) {
    const sessionId = path.split('/').filter(Boolean)[0];
    return sessionId ? [sessionId] : [];
  }

  const sessionIds = new Set();
  allPaths
    .filter((entry) => LAYER_DOMAINS.has(entry.domain) && entry.path)
    .forEach((entry) => sessionIds.add(entry.path.split('/').filter(Boolean)[0]));
  return Array.from(sessionIds);
}

function getLayerPaths(allPaths, layer, sessionId) {
  return allPaths.filter((entry) => entry.domain === layer && entry.path?.startsWith(`${sessionId}/`));
}

function groupByType(items, sessionId) {
  const groups = new Map();
  items.forEach((entry) => {
    const parts = entry.path.split('/').filter(Boolean);
    if (parts[0] !== sessionId) return;
    const type = parts[1] || 'misc';
    if (!groups.has(type)) groups.set(type, []);
    groups.get(type).push(entry);
  });

  return Array.from(groups.entries())
    .map(([type, groupItems]) => ({ type, items: groupItems, count: groupItems.length }))
    .sort((a, b) => b.count - a.count);
}

function visibleGroups(groups, activeType) {
  if (activeType) {
    const focused = groups.find((group) => group.type === activeType);
    return focused ? [focused] : [];
  }
  return groups.slice(0, 6);
}

function visibleItems(items, { activeDomain, activePath, sessionId, activeType, layer }) {
  if (!activeType) return items.slice(0, 3);

  const selected = items.find((entry) => entry.domain === activeDomain && entry.path === activePath);
  const isDeepSelection = activePath.split('/').filter(Boolean).length > 2;

  if (isDeepSelection) {
    if (selected) return [selected];

    const activeLeafKey = pathTailAfterType(activePath, sessionId, activeType);
    const matches = items.filter((entry) => pathTailAfterType(entry.path, sessionId, activeType) === activeLeafKey);
    return matches.slice(0, 3);
  }

  const sameBranch = items.filter((entry) => entry.path.startsWith(`${sessionId}/${activeType}/`));
  const picked = [];
  if (selected) picked.push(selected);
  sameBranch.forEach((entry) => {
    if (!picked.some((item) => item.domain === entry.domain && item.path === entry.path)) {
      picked.push(entry);
    }
  });

  if (!picked.length && layer === activeDomain) {
    const fallback = items.find((entry) => entry.path === activePath);
    if (fallback) picked.push(fallback);
  }

  return picked.slice(0, selected ? 6 : 4);
}

function pathTailAfterType(path, sessionId, type) {
  const prefix = `${sessionId}/${type}/`;
  return path.startsWith(prefix) ? path.slice(prefix.length) : path;
}
