export function parseJsonObject(value) {
  if (!value || typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) return null;
  try {
    const parsed = JSON.parse(trimmed);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function shortPath(path = '') {
  if (!path) return 'root';
  return path.split('/').filter(Boolean).pop() || path;
}

const SESSION_PREFIXES = [
  { prefix: 'cli:local_cli_user:', label: 'CLI Session' },
  { prefix: 'tui:local_tui_user:', label: 'TUI Session' },
  { prefix: 'interactive_cli:local_cli_user:', label: 'CLI Session' },
  { prefix: 'interactive_tui:local_tui_user:', label: 'TUI Session' },
];

function escapeRegExp(value = '') {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function formatSessionId(value = '') {
  const text = String(value || '');
  for (const { prefix, label } of SESSION_PREFIXES) {
    if (text.startsWith(prefix)) {
      return `${label} ${text.slice(prefix.length)}`;
    }
  }
  return text;
}

export function isSessionId(value = '') {
  const text = String(value || '');
  return SESSION_PREFIXES.some(({ prefix }) => text.startsWith(prefix));
}

export function formatDisplayText(value = '') {
  let text = value && typeof value === 'object'
    ? JSON.stringify(value, null, 2)
    : String(value ?? '');
  for (const { prefix, label } of SESSION_PREFIXES) {
    const pattern = new RegExp(`${escapeRegExp(prefix)}([A-Za-z0-9._-]+)`, 'g');
    text = text.replace(pattern, `${label} $1`);
  }
  return text;
}

export function formatPathParts(path = '') {
  const parts = String(path || '').split('/').filter(Boolean);
  if (!parts.length) return ['root'];
  return parts.map((part, index) => (index === 0 ? formatSessionId(part) : part));
}

export function formatPathLabel(path = '') {
  if (!path) return 'root';
  const parts = formatPathParts(path);
  return parts.length > 1 ? parts.join(' / ') : parts[0];
}

export function formatLeafLabel(path = '', fallback = '') {
  const parts = String(path || '').split('/').filter(Boolean);
  if (fallback && isSessionId(fallback)) return formatSessionId(fallback);
  if (!parts.length) return fallback || 'root';
  if (parts.length === 1 && isSessionId(parts[0])) return formatSessionId(parts[0]);
  return fallback || parts[parts.length - 1];
}

export function formatMemoryUri(domain = 'session', path = '') {
  const value = path || 'root';
  if (value === 'root') return `${domain}://root`;
  const parts = formatPathParts(value);
  if (!parts.length) return `${domain}://root`;
  return `${domain}://${parts.join('/')}`;
}

export function formatRawMemoryUri(domain = 'session', path = '') {
  return `${domain}://${path || 'root'}`;
}

export function formatNodeTitle(node = {}) {
  if (!node) return 'Memory node';
  return formatLeafLabel(node.path, node.name) || node.name || 'Memory node';
}
