import { AlertTriangle, ChevronRight, FileText, Folder, Link2 } from 'lucide-react';
import clsx from 'clsx';

import RankIndicator from './RankIndicator';
import { formatDisplayText, formatNodeTitle } from '../../../shared/utils/content';

export default function MemoryNodeCard({ node, currentDomain, onClick }) {
  const crossDomain = node.domain && node.domain !== currentDomain;
  const hasChildren = Number(node.approx_children_count || 0) > 0;
  const title = formatNodeTitle(node);

  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        'group flex h-full w-full flex-col rounded-lg border bg-white p-4 text-left shadow-sm transition hover:-translate-y-0.5 hover:shadow-md',
        crossDomain ? 'border-violet-200 hover:border-violet-300' : 'border-slate-200 hover:border-cyan-300',
      )}
    >
      <div className="mb-3 flex w-full items-start gap-3">
        <div
          className={clsx(
            'flex h-10 w-10 shrink-0 items-center justify-center rounded-md border',
            hasChildren ? 'border-cyan-200 bg-cyan-50 text-cyan-700' : 'border-slate-200 bg-slate-50 text-slate-500',
          )}
        >
          {hasChildren ? <Folder size={18} /> : <FileText size={18} />}
        </div>

        <div className="min-w-0 flex-1">
          <h3 className="line-clamp-2 text-sm font-semibold leading-5 text-slate-900">{title}</h3>
          {crossDomain ? (
            <span className="mt-1 inline-flex items-center gap-1 rounded-md border border-violet-200 bg-violet-50 px-1.5 py-0.5 font-mono text-[10px] text-violet-700">
              <Link2 size={9} />
              {node.domain}://
            </span>
          ) : null}
        </div>

        <RankIndicator priority={node.priority} />
      </div>

      {node.disclosure ? (
        <div className="mb-3 flex items-start gap-1.5 rounded-md border border-amber-200 bg-amber-50 px-2 py-1.5 text-[11px] leading-4 text-amber-800">
          <AlertTriangle size={12} className="mt-0.5 shrink-0" />
          <span className="line-clamp-2">{formatDisplayText(node.disclosure)}</span>
        </div>
      ) : null}

      <p className={clsx('line-clamp-3 flex-1 text-xs leading-5', node.content_snippet ? 'text-slate-600' : 'italic text-slate-400')}>
        {node.content_snippet ? formatDisplayText(node.content_snippet) : 'No preview available'}
      </p>

      <div className="mt-4 flex justify-end text-cyan-700 opacity-0 transition group-hover:translate-x-1 group-hover:opacity-100">
        <ChevronRight size={16} />
      </div>
    </button>
  );
}
