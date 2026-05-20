export default function EmptyState({ icon: Icon, title, detail, action }) {
  return (
    <div className="flex h-full min-h-64 flex-col items-center justify-center rounded-md border border-dashed border-slate-300 bg-white p-8 text-center">
      {Icon ? <Icon size={30} className="mb-3 text-slate-400" /> : null}
      <div className="text-sm font-medium text-slate-700">{title}</div>
      {detail ? <div className="mt-1 max-w-md text-sm text-slate-500">{detail}</div> : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}
