import clsx from 'clsx';

const variants = {
  primary: 'bg-cyan-600 text-white hover:bg-cyan-500 disabled:bg-slate-200 disabled:text-slate-400',
  secondary: 'border border-slate-300 bg-white text-slate-700 hover:border-slate-400 hover:bg-slate-50',
  danger: 'border border-rose-300 bg-rose-50 text-rose-700 hover:bg-rose-100',
  ghost: 'text-slate-500 hover:bg-slate-100 hover:text-slate-950',
};

export default function Button({
  children,
  className,
  icon: Icon,
  variant = 'secondary',
  type = 'button',
  ...props
}) {
  return (
    <button
      type={type}
      className={clsx(
        'inline-flex h-9 items-center justify-center gap-2 rounded-md px-3 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-60',
        variants[variant],
        className,
      )}
      {...props}
    >
      {Icon ? <Icon size={16} /> : null}
      {children}
    </button>
  );
}
