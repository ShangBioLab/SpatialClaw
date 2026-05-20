import assert from 'node:assert/strict';

import {
  formatDisplayText,
  formatLeafLabel,
  formatMemoryUri,
  formatNodeTitle,
  formatPathLabel,
  formatSessionId,
} from './content.js';
import {
  buildSessionNavigation,
  layerTypeChildren,
  preferredLayeredViewMode,
  shouldAutoRevealForRouteChange,
  visibleChildCount,
} from '../../features/memory/utils/sessionNavigation.js';

assert.equal(formatSessionId('cli:local_cli_user:dc5b74e9'), 'CLI Session dc5b74e9');
assert.equal(formatSessionId('tui:local_tui_user:abc123'), 'TUI Session abc123');
assert.equal(formatSessionId('interactive_cli:local_cli_user:old1'), 'CLI Session old1');

assert.equal(
  formatPathLabel('cli:local_cli_user:dc5b74e9/analysis/run-1'),
  'CLI Session dc5b74e9 / analysis / run-1',
);
assert.equal(
  formatMemoryUri('episodic', 'cli:local_cli_user:dc5b74e9/analysis/run-1'),
  'episodic://CLI Session dc5b74e9/analysis/run-1',
);
assert.equal(
  formatLeafLabel('cli:local_cli_user:dc5b74e9', 'cli:local_cli_user:dc5b74e9'),
  'CLI Session dc5b74e9',
);
assert.equal(
  formatNodeTitle({ path: 'cli:local_cli_user:dc5b74e9', name: 'cli:local_cli_user:dc5b74e9' }),
  'CLI Session dc5b74e9',
);
assert.equal(
  formatDisplayText({ session_id: 'cli:local_cli_user:dc5b74e9', nested: { user_id: 'tui:local_tui_user:t01' } }),
  '{\n  "session_id": "CLI Session dc5b74e9",\n  "nested": {\n    "user_id": "TUI Session t01"\n  }\n}',
);

const navigation = buildSessionNavigation([
  { domain: 'session', path: 'cli:local_cli_user:s1', name: 'cli:local_cli_user:s1' },
  { domain: 'episodic', path: 'cli:local_cli_user:s1' },
  { domain: 'episodic', path: 'cli:local_cli_user:s1/analysis' },
  { domain: 'episodic', path: 'cli:local_cli_user:s1/analysis/run1' },
  { domain: 'semantic', path: 'cli:local_cli_user:s1' },
  { domain: 'semantic', path: 'cli:local_cli_user:s1/preference' },
  { domain: 'semantic', path: 'cli:local_cli_user:s1/preference/global' },
  { domain: 'semantic', path: 'cli:local_cli_user:s2/insight/domain_1' },
]);

assert.equal(navigation.length, 2);
assert.equal(navigation[0].id, 'cli:local_cli_user:s1');
assert.equal(navigation[0].layers.episodic.pathCount, 3);
assert.equal(navigation[0].layers.episodic.typeCounts.analysis, 2);
assert.equal(navigation[0].layers.semantic.typeCounts.preference, 2);
assert.equal(navigation[1].id, 'cli:local_cli_user:s2');
assert.equal(navigation[1].sessionNode, null);
assert.equal(navigation[1].layers.semantic.exists, false);
assert.deepEqual(layerTypeChildren(navigation[1].layers.semantic), [
  {
    domain: 'semantic',
    path: 'cli:local_cli_user:s2/insight',
    name: 'insight',
    approx_children_count: 1,
    is_virtual: true,
  },
]);

assert.equal(
  shouldAutoRevealForRouteChange(
    { domain: 'episodic', path: 'cli:local_cli_user:s1/analysis/run1' },
    { domain: 'episodic', path: 'cli:local_cli_user:s1/analysis/run1' },
    true,
    false,
  ),
  false,
);
assert.equal(
  shouldAutoRevealForRouteChange(
    { domain: 'episodic', path: 'cli:local_cli_user:s1/analysis/run1' },
    { domain: 'semantic', path: 'cli:local_cli_user:s2/preference' },
    true,
    false,
  ),
  true,
);
assert.equal(
  shouldAutoRevealForRouteChange(
    { domain: 'episodic', path: 'cli:local_cli_user:s1/analysis/run1' },
    { domain: 'semantic', path: 'cli:local_cli_user:s2/preference' },
    true,
    true,
  ),
  false,
);

assert.equal(visibleChildCount(undefined), undefined);
assert.equal(visibleChildCount(0), undefined);
assert.equal(visibleChildCount('0'), undefined);
assert.equal(visibleChildCount(2), 2);

assert.equal(
  preferredLayeredViewMode({
    domain: 'semantic',
    path: 'cli:local_cli_user:s1/preference/global/language',
    node: {
      is_virtual: false,
      content: '{"key":"language","value":"Chinese"}',
    },
    children: [],
  }),
  'detail',
);
assert.equal(
  preferredLayeredViewMode({
    domain: 'episodic',
    path: 'cli:local_cli_user:s1/analysis/run-1',
    node: { is_virtual: false, content: '{"event":"started analysis"}' },
    children: [],
  }),
  'detail',
);
assert.equal(
  preferredLayeredViewMode({
    domain: 'session',
    path: 'cli:local_cli_user:s1',
    node: { is_virtual: false, content: '{"session_id":"cli:local_cli_user:s1"}' },
    children: [],
  }),
  'detail',
);
assert.equal(
  preferredLayeredViewMode({
    domain: 'semantic',
    path: 'cli:local_cli_user:s1/preference/global',
    node: {
      is_virtual: false,
      content: 'Container node: semantic://cli:local_cli_user:s1/preference/global',
    },
    children: [{ path: 'cli:local_cli_user:s1/preference/global/language' }],
  }),
  'graph',
);
assert.equal(
  preferredLayeredViewMode({
    domain: 'dataset',
    path: 'sample',
    node: { is_virtual: false, content: 'Sample' },
    children: [],
  }),
  null,
);
assert.equal(
  preferredLayeredViewMode({
    domain: 'semantic',
    path: 'cli:local_cli_user:s1/preference/global/language',
    node: { is_virtual: false, content: '{"value":"Chinese"}' },
    children: [],
    editing: true,
  }),
  null,
);

console.log('content formatter tests passed');
