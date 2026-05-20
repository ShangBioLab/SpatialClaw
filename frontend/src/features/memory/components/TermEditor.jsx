import { useEffect, useRef, useState } from 'react';
import { Plus, Save, Tag, X } from 'lucide-react';

import { addGlossary, removeGlossary } from '../../../lib/memoryApi';

export default function TermEditor({ keywords = [], nodeUuid, onUpdate }) {
  const [draft, setDraft] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    if (isAdding) inputRef.current?.focus();
  }, [isAdding]);

  function cancelDraft() {
    setDraft('');
    setIsAdding(false);
  }

  async function submitDraft() {
    const keyword = draft.trim();
    if (!keyword || !nodeUuid) return;

    try {
      await addGlossary(keyword, nodeUuid);
      cancelDraft();
      onUpdate?.();
    } catch (err) {
      window.alert(`Failed to add keyword: ${err.response?.data?.detail || err.message}`);
    }
  }

  async function removeKeyword(keyword) {
    if (!nodeUuid) return;

    try {
      await removeGlossary(keyword, nodeUuid);
      onUpdate?.();
    } catch (err) {
      window.alert(`Failed to remove keyword: ${err.response?.data?.detail || err.message}`);
    }
  }

  function handleDraftKeyDown(event) {
    if (event.key === 'Enter') submitDraft();
    if (event.key === 'Escape') cancelDraft();
  }

  return (
    <div className="flex max-w-full items-start gap-2 text-xs text-slate-500">
      <Tag size={14} className="mt-0.5 shrink-0 text-amber-600" />
      <div className="min-w-0 flex flex-wrap items-center gap-1.5">
        <span className="font-medium text-amber-700">Glossary</span>

        {keywords.map((keyword) => (
          <span
            key={keyword}
            className="inline-flex items-center gap-1 rounded-md border border-amber-200 bg-amber-50 px-2 py-0.5 font-mono text-[11px] text-amber-800"
          >
            {keyword}
            <button
              type="button"
              aria-label={`Remove ${keyword}`}
              className="rounded text-amber-500 transition hover:text-rose-600"
              onClick={() => removeKeyword(keyword)}
            >
              <X size={10} />
            </button>
          </span>
        ))}

        {isAdding ? (
          <span className="inline-flex items-center gap-1">
            <input
              ref={inputRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={handleDraftKeyDown}
              onBlur={() => {
                if (!draft.trim()) cancelDraft();
              }}
              placeholder="keyword"
              className="h-6 w-32 rounded-md border border-amber-200 bg-white px-2 font-mono text-[11px] text-slate-800 outline-none transition focus:border-amber-400 focus:ring-2 focus:ring-amber-100"
            />
            <button
              type="button"
              aria-label="Save keyword"
              className="rounded-md border border-amber-200 bg-white p-1 text-amber-700 transition hover:bg-amber-50"
              onClick={submitDraft}
            >
              <Save size={11} />
            </button>
          </span>
        ) : (
          <button
            type="button"
            className="inline-flex h-6 items-center gap-1 rounded-md border border-dashed border-amber-300 bg-white px-2 text-[11px] font-medium text-amber-700 transition hover:bg-amber-50"
            onClick={() => setIsAdding(true)}
          >
            <Plus size={10} />
            Add
          </button>
        )}
      </div>
    </div>
  );
}
