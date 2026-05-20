import clsx from 'clsx';

const toneClasses = {
  neutral: 'border-slate-300 bg-slate-50 text-slate-700',
  cyan: 'border-cyan-200 bg-cyan-50 text-cyan-700',
  green: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  amber: 'border-amber-200 bg-amber-50 text-amber-700',
  rose: 'border-rose-200 bg-rose-50 text-rose-700',
};

export default function StatusPill({ children, tone = 'neutral', className }) {
  return (
    <span className={clsx('inline-flex h-6 items-center rounded-md border px-2 text-xs font-medium', toneClasses[tone], className)}>
      {children}
    </span>
  );
}
