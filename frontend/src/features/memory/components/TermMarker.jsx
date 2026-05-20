import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { BookOpen, X } from 'lucide-react';
import clsx from 'clsx';

import { formatDisplayText, formatMemoryUri } from '../../../shared/utils/content';

function glossaryMatches(content, glossary = []) {
  if (!content || !glossary.length) return [];

  const candidates = glossary.flatMap((entry) => {
    if (!entry.keyword) return [];
    const matches = [];
    let start = content.indexOf(entry.keyword);
    while (start !== -1) {
      matches.push({
        start,
        end: start + entry.keyword.length,
        keyword: entry.keyword,
        nodes: entry.nodes || [],
      });
      start = content.indexOf(entry.keyword, start + entry.keyword.length);
    }
    return matches;
  });

  return candidates
    .sort((left, right) => left.start - right.start || right.end - right.start - (left.end - left.start))
    .reduce((accepted, match) => {
      const previous = accepted[accepted.length - 1];
      if (!previous || match.start >= previous.end) accepted.push(match);
      return accepted;
    }, []);
}

function splitContent(content, matches) {
  const parts = [];
  let cursor = 0;

  for (const match of matches) {
    if (match.start > cursor) {
      parts.push({ text: content.slice(cursor, match.start) });
    }
    parts.push({ text: content.slice(match.start, match.end), match });
    cursor = match.end;
  }

  if (cursor < content.length) {
    parts.push({ text: content.slice(cursor) });
  }

  return parts;
}

function popupPosition(rect) {
  const width = 288;
  const margin = 16;
  const left = Math.max(margin, Math.min(rect.left, window.innerWidth - width - margin));
  const useTop = rect.bottom + 260 <= window.innerHeight || rect.top < 280;

  return useTop
    ? { left, top: rect.bottom + 6, maxHeight: window.innerHeight - rect.bottom - margin }
    : { left, bottom: window.innerHeight - rect.top + 6, maxHeight: rect.top - margin };
}

function GlossaryPopup({ keyword, nodes, position, onClose, onNavigate }) {
  const ref = useRef(null);

  useEffect(() => {
    function handlePointerDown(event) {
      if (ref.current && !ref.current.contains(event.target)) onClose();
    }

    document.addEventListener('mousedown', handlePointerDown);
    return () => document.removeEventListener('mousedown', handlePointerDown);
  }, [onClose]);

  return createPortal(
    <div
      ref={ref}
      className="fixed z-[100] flex w-72 flex-col overflow-hidden rounded-lg border border-amber-200 bg-white shadow-xl shadow-slate-200/80"
      style={position}
    >
      <div className="flex shrink-0 items-center gap-2 border-b border-slate-200 px-3 py-2">
        <BookOpen size={13} className="text-amber-600" />
        <span className="truncate text-xs font-semibold text-amber-800">{keyword}</span>
        <button
          type="button"
          aria-label="Close glossary popup"
          className="ml-auto rounded p-1 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
          onClick={onClose}
        >
          <X size={12} />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {nodes.map((node, index) => (
          <GlossaryTarget
            key={node.uri || index}
            node={node}
            onNavigate={onNavigate}
            onClose={onClose}
          />
        ))}
      </div>
    </div>,
    document.body,
  );
}

function GlossaryTarget({ node, onNavigate, onClose }) {
  const isUnlinked = node.uri?.startsWith('unlinked://');

  function openNode() {
    if (isUnlinked) return;
    const match = node.uri?.match(/^([^:]+):\/\/(.*)$/);
    if (match) onNavigate(match[2], match[1]);
    onClose();
  }

  return (
    <button
      type="button"
      className={clsx(
        'w-full rounded-md px-2.5 py-2 text-left transition',
        isUnlinked ? 'cursor-default bg-slate-50 opacity-80' : 'hover:bg-amber-50',
      )}
      onClick={openNode}
    >
      <div className="flex items-center justify-between gap-2">
        <code className={clsx('block flex-1 truncate font-mono text-[11px]', isUnlinked ? 'text-slate-500' : 'text-indigo-700')}>
          {formatGlossaryUri(node.uri)}
        </code>
        {isUnlinked ? (
          <span className="shrink-0 rounded border border-rose-200 bg-rose-50 px-1.5 py-0.5 text-[9px] text-rose-700">
            Orphaned
          </span>
        ) : null}
      </div>
      {node.content_snippet ? (
        <p className="mt-1 line-clamp-2 text-[10px] leading-snug text-slate-500">
          {formatDisplayText(node.content_snippet)}
        </p>
      ) : null}
    </button>
  );
}

function formatGlossaryUri(uri = '') {
  const match = String(uri || '').match(/^([^:]+):\/\/(.*)$/);
  if (!match) return formatDisplayText(uri);
  if (match[1] === 'unlinked') return uri;
  return formatMemoryUri(match[1], match[2]);
}

export default function TermMarker({ content = '', glossary = [], currentNodeUuid, onNavigate }) {
  const [popup, setPopup] = useState(null);

  useEffect(() => setPopup(null), [content]);

  const linkedGlossary = useMemo(() => (
    glossary
      .map((entry) => ({
        ...entry,
        nodes: (entry.nodes || []).filter((node) => node.node_uuid !== currentNodeUuid),
      }))
      .filter((entry) => entry.nodes.length > 0)
  ), [glossary, currentNodeUuid]);

  const matches = useMemo(() => glossaryMatches(content, linkedGlossary), [content, linkedGlossary]);
  const parts = useMemo(() => splitContent(content, matches), [content, matches]);

  function showPopup(event, match) {
    setPopup({
      keyword: match.keyword,
      nodes: match.nodes,
      position: popupPosition(event.currentTarget.getBoundingClientRect()),
    });
  }

  return (
    <div className="relative">
      <pre className="whitespace-pre-wrap font-serif leading-7 text-slate-700">
        {parts.map((part, index) => (
          part.match ? (
            <button
              key={`${part.text}-${index}`}
              type="button"
              className="font-serif text-amber-700 underline decoration-amber-400 decoration-dotted underline-offset-2 transition hover:text-amber-900"
              onClick={(event) => showPopup(event, part.match)}
            >
              {part.text}
            </button>
          ) : (
            <Fragment key={`${part.text}-${index}`}>{part.text}</Fragment>
          )
        ))}
      </pre>

      {popup ? (
        <GlossaryPopup
          keyword={popup.keyword}
          nodes={popup.nodes}
          position={popup.position}
          onClose={() => setPopup(null)}
          onNavigate={onNavigate}
        />
      ) : null}
    </div>
  );
}
