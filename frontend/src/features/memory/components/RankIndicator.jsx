import { Star } from 'lucide-react';
import clsx from 'clsx';

const PRIORITY_STYLES = [
  { test: (value) => value === 0, className: 'border-rose-200 bg-rose-50 text-rose-700' },
  { test: (value) => value <= 2, className: 'border-amber-200 bg-amber-50 text-amber-700' },
  { test: (value) => value <= 5, className: 'border-sky-200 bg-sky-50 text-sky-700' },
  { test: () => true, className: 'border-slate-300 bg-slate-50 text-slate-700' },
];

const SIZE_STYLES = {
  sm: 'gap-1 px-1.5 py-0.5 text-[10px]',
  lg: 'gap-1.5 px-2.5 py-1 text-xs',
};

export default function RankIndicator({ priority, size = 'sm' }) {
  if (priority === null || priority === undefined) return null;

  const tone = PRIORITY_STYLES.find((entry) => entry.test(priority))?.className;
  const iconSize = size === 'lg' ? 12 : 9;

  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-md border font-mono font-semibold',
        tone,
        SIZE_STYLES[size] || SIZE_STYLES.sm,
      )}
      title={`Priority ${priority}`}
    >
      <Star size={iconSize} />
      {priority}
    </span>
  );
}
