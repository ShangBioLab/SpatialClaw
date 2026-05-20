export default function PageHeader({ title, eyebrow, actions, meta }) {
  return (
    <header className="flex shrink-0 flex-col gap-3 border-b border-slate-200 bg-white/95 px-5 py-4 md:flex-row md:items-center md:justify-between">
      <div className="min-w-0">
        {eyebrow ? <div className="mb-1 text-xs font-medium uppercase tracking-wider text-cyan-700">{eyebrow}</div> : null}
        <h1 className="truncate text-lg font-semibold text-slate-950">{title}</h1>
        {meta ? <div className="mt-1 text-sm text-slate-500">{meta}</div> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </header>
  );
}
