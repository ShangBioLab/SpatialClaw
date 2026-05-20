const LAYER_DOMAINS = new Set(['episodic', 'semantic']);
const LAYERED_DOMAINS = new Set(['session', 'episodic', 'semantic']);

function firstPathPart(path = '') {
  return String(path || '').split('/').filter(Boolean)[0] || '';
}

function memoryTypeFromPath(path = '', sessionId = '') {
  const parts = String(path || '').split('/').filter(Boolean);
  if (parts[0] !== sessionId) return '';
  return parts[1] || '';
}

function makeSession(sessionId) {
  return {
    id: sessionId,
    sessionNode: null,
    layers: {
      episodic: { domain: 'episodic', rootPath: sessionId, exists: false, pathCount: 0, typeCounts: {} },
      semantic: { domain: 'semantic', rootPath: sessionId, exists: false, pathCount: 0, typeCounts: {} },
    },
  };
}

export function buildSessionNavigation(paths = []) {
  const sessions = new Map();

  function ensureSession(sessionId) {
    if (!sessionId) return null;
    if (!sessions.has(sessionId)) sessions.set(sessionId, makeSession(sessionId));
    return sessions.get(sessionId);
  }

  paths.forEach((entry) => {
    const domain = entry?.domain || '';
    const path = entry?.path || '';
    const sessionId = firstPathPart(path);
    const session = ensureSession(sessionId);
    if (!session) return;

    if (domain === 'session' && path === sessionId) {
      session.sessionNode = entry;
      return;
    }

    if (!LAYER_DOMAINS.has(domain)) return;

    const layer = session.layers[domain];
    layer.pathCount += 1;
    if (path === sessionId) layer.exists = true;

    const type = memoryTypeFromPath(path, sessionId);
    if (type) layer.typeCounts[type] = (layer.typeCounts[type] || 0) + 1;
  });

  return Array.from(sessions.values())
    .map((session) => ({
      ...session,
      totalPathCount: session.layers.episodic.pathCount + session.layers.semantic.pathCount,
    }))
    .sort((left, right) => left.id.localeCompare(right.id));
}

export function layerTypeChildren(layer) {
  return Object.entries(layer?.typeCounts || {})
    .sort(([leftType, leftCount], [rightType, rightCount]) => rightCount - leftCount || leftType.localeCompare(rightType))
    .map(([type, count]) => ({
      domain: layer.domain,
      path: `${layer.rootPath}/${type}`,
      name: type,
      approx_children_count: count,
      is_virtual: !layer.exists,
    }));
}

export function shouldAutoRevealForRouteChange(previousRoute, currentRoute, targeted, expanded) {
  const changed = previousRoute?.domain !== currentRoute?.domain || previousRoute?.path !== currentRoute?.path;
  return Boolean(changed && targeted && !expanded);
}

export function visibleChildCount(count) {
  const value = Number(count);
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

export function preferredLayeredViewMode({ domain, path, node, children = [], editing = false }) {
  if (editing || !LAYERED_DOMAINS.has(domain)) return null;

  const hasPath = Boolean(path);
  const hasChildren = Array.isArray(children) && children.length > 0;
  const hasConcreteContent = Boolean(node && !node.is_virtual && node.content);

  return hasPath && hasConcreteContent && !hasChildren ? 'detail' : 'graph';
}
