import type { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`card p-4 ${className}`}>{children}</div>;
}

export function SectionTitle({ children, hint }: { children: ReactNode; hint?: string }) {
  return (
    <div className="mb-2 flex items-baseline justify-between">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-soft">{children}</h2>
      {hint && <span className="text-xs text-ink-faint">{hint}</span>}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "secondary",
  disabled,
  title,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "danger";
  disabled?: boolean;
  title?: string;
  className?: string;
}) {
  return (
    <button
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`btn btn-${variant} ${className}`}
    >
      {children}
    </button>
  );
}

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={`input ${props.className ?? ""}`} />;
}

export function Select(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={`input ${props.className ?? ""}`} />;
}

const STATUS_COLORS: Record<string, string> = {
  completed: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  failed: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
  cancelled: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300",
  approved: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  validated: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  prepared: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  training: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
  running: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
  queued: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
  draft: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300",
  raw: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300",
};

export function StatusBadge({ status, label }: { status: string; label?: string }) {
  const color = STATUS_COLORS[status] ?? "bg-blue-50 text-ink-soft dark:bg-[#2a3040] dark:text-[#9fb2d1]";
  return <span className={`badge ${color}`}>{label ?? status}</span>;
}

export function ErrorBox({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300">
      {message}
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="card flex flex-col items-center gap-2 px-6 py-10 text-center">
      <div className="text-sm font-medium text-ink">{title}</div>
      {hint && <div className="max-w-md text-xs text-ink-faint">{hint}</div>}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-ink-faint">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-ink-faint border-t-transparent" />
      {label ?? "Yükleniyor..."}
    </div>
  );
}

export function KeyValue({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="flex justify-between gap-4 border-b border-black/5 py-1 text-sm last:border-0 dark:border-white/5">
      <span className="text-ink-faint">{k}</span>
      <span className="text-right">{v}</span>
    </div>
  );
}
