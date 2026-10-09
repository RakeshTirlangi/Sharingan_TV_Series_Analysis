import clsx from "clsx";
import { BookOpen } from "lucide-react";
import { motion } from "motion/react";
import { useScrollSpy } from "../lib/hooks";
import { SharinganEye } from "./SharinganEye";

const LINKS = [
  { id: "overview", label: "Overview" },
  { id: "themes", label: "Themes" },
  { id: "network", label: "Network" },
  { id: "jutsu", label: "Jutsu" },
  { id: "chat", label: "Chat" },
];
const IDS = LINKS.map((l) => l.id);

export function Navbar() {
  const active = useScrollSpy(IDS);
  return (
    <motion.nav
      initial={{ y: -40, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.6 }}
      className="fixed inset-x-0 top-0 z-50 border-b border-line bg-ink/70 backdrop-blur-xl"
    >
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6">
        <a href="#overview" className="flex items-center gap-2.5">
          <SharinganEye className="h-8 w-8" spin="medium" glow={false} />
          <span className="font-display text-lg font-semibold tracking-tight">Sharingan</span>
        </a>
        <div className="hidden items-center gap-1 rounded-full border border-line bg-white/[0.02] p-1 md:flex">
          {LINKS.map((l) => (
            <a key={l.id} href={`#${l.id}`}
               className={clsx("relative rounded-full px-4 py-1.5 text-sm transition-colors",
                               active === l.id ? "text-fg" : "text-fg-3 hover:text-fg")}>
              {active === l.id && (
                <motion.span layoutId="nav-pill" className="absolute inset-0 rounded-full bg-white/10"
                             transition={{ type: "spring", bounce: 0.2, duration: 0.5 }} />
              )}
              <span className="relative">{l.label}</span>
            </a>
          ))}
        </div>
        <a href="/docs" className="inline-flex items-center gap-2 rounded-full border border-line px-3.5 py-1.5 text-sm text-fg-2 transition hover:border-white/20 hover:text-fg">
          <BookOpen className="h-4 w-4" /> <span className="hidden sm:inline">API docs</span>
        </a>
      </div>
      <div className="flex gap-1 overflow-x-auto border-t border-line px-3 py-2 md:hidden">
        {LINKS.map((l) => (
          <a key={l.id} href={`#${l.id}`}
             className={clsx("shrink-0 rounded-full px-3 py-1 text-xs transition-colors",
                             active === l.id ? "bg-white/10 text-fg" : "text-fg-3")}>
            {l.label}
          </a>
        ))}
      </div>
    </motion.nav>
  );
}
