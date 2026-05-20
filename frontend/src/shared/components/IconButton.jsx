import clsx from 'clsx';

export default function IconButton({ icon: Icon, label, className, ...props }) {
  return (
    <button
      aria-label={label}
      title={label}
      className={clsx(
        'inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-300 bg-white text-slate-500 transition hover:border-cyan-300 hover:bg-cyan-50 hover:text-cyan-700',
        className,
      )}
      {...props}
    >
      <Icon size={16} />
    </button>
  );
}
