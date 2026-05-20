export default function LoadingState({ label = 'Loading' }) {
  return (
    <div className="flex items-center gap-3 text-sm text-slate-500">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-200 border-t-cyan-600" />
      <span>{label}</span>
    </div>
  );
}
