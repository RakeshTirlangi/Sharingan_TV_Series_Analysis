import clsx from "clsx";
import { AlertTriangle, Loader2, Terminal } from "lucide-react";
import { animate, motion, useInView, useMotionValue, useReducedMotion, useTransform } from "motion/react";
import { type ReactNode, useEffect, useRef } from "react";
import { fmt } from "../lib/hooks";

/** Scroll-reveal props; reduced-motion users get content immediately, with no fade. */
function useReveal(y: number) {
  const reduce = useReducedMotion();
  return reduce ? { initial: false as const } : { initial: { opacity: 0, y } };
}

export function Section({ id, eyebrow, title, lead, children }: {
  id: string; eyebrow: string; title: ReactNode; lead?: ReactNode; children: ReactNode;
}) {
  const reveal = useReveal(24);
  return (
    <section id={id} className="relative mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:py-28">
      <motion.header
        {...reveal} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }} className="mb-10 max-w-3xl"
      >
        <p className="mb-3 font-mono text-xs uppercase tracking-[0.25em] text-blood-400">{eyebrow}</p>
        <h2 className="font-display text-3xl font-semibold tracking-tight sm:text-5xl">{title}</h2>
        {lead && <p className="mt-4 text-base leading-relaxed text-fg-2 sm:text-lg">{lead}</p>}
      </motion.header>
      {children}
    </section>
  );
}

export function Card({ className, children, title, subtitle, actions }: {
  className?: string; children: ReactNode; title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode;
}) {
  const reveal = useReveal(18);
  return (
    <motion.div
      {...reveal} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
      className={clsx("glass grain rounded-2xl p-5 sm:p-6", className)}
    >
      {(title || actions) && (
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            {title && <h3 className="font-display text-lg font-semibold">{title}</h3>}
            {subtitle && <p className="mt-1 text-sm text-fg-3">{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </motion.div>
  );
}

export function Pill({ active, children, onClick, color, disabled }: {
  active?: boolean; children: ReactNode; onClick?: () => void; color?: string; disabled?: boolean;
}) {
  return (
    <button
      type="button" onClick={onClick} disabled={disabled}
      className={clsx(
        "inline-flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-sm transition-all duration-200",
        "disabled:cursor-not-allowed disabled:opacity-40",
        active ? "border-white/25 bg-white/10 text-fg shadow-[0_0_0_1px_rgba(255,255,255,0.05)]"
               : "border-line text-fg-2 hover:border-white/20 hover:text-fg",
      )}
    >
      {color && <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />}
      {children}
    </button>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-3 py-16 text-fg-3">
      <Loader2 className="h-5 w-5 animate-spin text-blood-400" /> <span className="text-sm">{label}…</span>
    </div>
  );
}

/** Shown when a stage has not been computed or the API is unreachable. */
export function NotReady({ error, command }: { error?: Error; command: string }) {
  const offline = error && !("status" in error);
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-line px-6 py-12 text-center">
      <AlertTriangle className="h-6 w-6 text-blood-400" />
      <p className="max-w-md text-sm text-fg-2">
        {offline ? "Can't reach the Sharingan API." : (error?.message ?? "This stage hasn't been computed yet.")}
      </p>
      <code className="inline-flex items-center gap-2 rounded-lg bg-surface-2 px-3 py-1.5 font-mono text-xs text-fg-2">
        <Terminal className="h-3.5 w-3.5" /> {offline ? "sharingan app" : command}
      </code>
    </div>
  );
}

export function CountUp({ value, className }: { value: number; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const mv = useMotionValue(0);
  const text = useTransform(mv, (v) => fmt.format(Math.round(v)));
  const reduce = useReducedMotion();
  useEffect(() => {
    if (reduce) { mv.set(value); return; }
    if (inView) { const c = animate(mv, value, { duration: 1.6, ease: [0.16, 1, 0.3, 1] }); return () => c.stop(); }
  }, [inView, value, mv, reduce]);
  return <motion.span ref={ref} className={className}>{text}</motion.span>;
}

export function Tooltip({ children }: { children: ReactNode }) {
  return <div className="rounded-lg border border-line bg-surface/95 px-3 py-2 text-xs shadow-2xl backdrop-blur">{children}</div>;
}
