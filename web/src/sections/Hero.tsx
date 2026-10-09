import { ArrowDown, Brain, Network, ScrollText, Sparkles, Swords } from "lucide-react";
import { motion } from "motion/react";
import { CountUp } from "../components/ui";
import { SharinganEye } from "../components/SharinganEye";
import type { Overview } from "../lib/api";

const STAGES = [
  { icon: ScrollText, title: "Scrape", body: "Scrapy spider on the MediaWiki API: 2.7k jutsu articles in about 60 requests.", key: null },
  { icon: Sparkles, title: "Themes", body: "Zero-shot NLI scores every dialogue chunk against each theme.", key: "themes" },
  { icon: Network, title: "Network", body: "spaCy NER + a cast gazetteer → weighted co-occurrence graph.", key: "ner" },
  { icon: Swords, title: "Classify", body: "Fine-tuned encoder vs. TF-IDF and embedding baselines.", key: "jutsu" },
  { icon: Brain, title: "Chat", body: "Retrieval-augmented LLM personas grounded in real lines.", key: "persona" },
] as const;

export function Hero({ overview }: { overview?: Overview }) {
  const stats = [
    { label: "Episodes", value: overview?.episodes ?? 220 },
    { label: "Dialogue lines", value: overview?.lines ?? 0 },
    { label: "Jutsu scraped", value: overview?.jutsus ?? 0 },
    { label: "Characters", value: overview?.characters ?? 0 },
    { label: "Relationships", value: overview?.relationships ?? 0 },
  ];
  return (
    <section id="overview" className="relative overflow-hidden pt-28 sm:pt-32">
      <div className="mx-auto grid max-w-7xl items-center gap-12 px-4 sm:px-6 lg:grid-cols-[1.15fr_1fr]">
        <div>
          <motion.p initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
                    className="mb-5 inline-flex items-center gap-2 rounded-full border border-blood-500/30 bg-blood-500/10 px-3 py-1 font-mono text-xs uppercase tracking-[0.2em] text-blood-300">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blood-400" /> NLP · LLMs · Naruto
          </motion.p>
          <motion.h1 initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
                     transition={{ delay: 0.2, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                     className="font-display text-5xl font-bold leading-[1.02] tracking-tight sm:text-7xl">
            The eye that <span className="text-gradient-blood">reads</span> a whole series.
          </motion.h1>
          <motion.p initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35, duration: 0.8 }}
                    className="mt-6 max-w-xl text-lg leading-relaxed text-fg-2">
            Sharingan runs all 220 episodes of <em>Naruto</em> through modern NLP. It traces themes across story
            arcs, maps who matters to whom, classifies jutsu, and copies characters closely enough to hold a
            conversation with them.
          </motion.p>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="mt-8 flex flex-wrap gap-3">
            <a href="#themes" className="group inline-flex items-center gap-2 rounded-full bg-blood-500 px-5 py-2.5 font-medium text-white shadow-[0_8px_30px_-6px_rgba(229,56,59,0.6)] transition hover:bg-blood-400">
              Explore the analysis <ArrowDown className="h-4 w-4 transition group-hover:translate-y-0.5" />
            </a>
            <a href="#chat" className="inline-flex items-center gap-2 rounded-full border border-line px-5 py-2.5 font-medium text-fg-2 transition hover:border-white/25 hover:text-fg">
              Talk to Naruto
            </a>
          </motion.div>
        </div>
        <motion.div initial={{ opacity: 0, scale: 0.85, rotate: -30 }} animate={{ opacity: 1, scale: 1, rotate: 0 }}
                    transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1] }} className="relative mx-auto w-full max-w-md">
          <div className="absolute inset-[-12%] rounded-full border border-blood-500/10" />
          <div className="absolute inset-[-24%] rounded-full border border-blood-500/5" />
          <SharinganEye className="aspect-square w-full" />
        </motion.div>
      </div>

      <div className="mx-auto mt-20 max-w-7xl px-4 sm:px-6">
        <div className="glass grid grid-cols-2 divide-line rounded-2xl sm:grid-cols-5 sm:divide-x">
          {stats.map((s) => (
            <div key={s.label} className="px-5 py-6 text-center">
              <CountUp value={s.value} className="font-display text-3xl font-semibold tabular-nums sm:text-4xl" />
              <p className="mt-1 text-xs uppercase tracking-widest text-fg-3">{s.label}</p>
            </div>
          ))}
        </div>

        <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {STAGES.map((st, i) => (
            <motion.div key={st.title} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }} transition={{ delay: 0.08 * i, duration: 0.5 }}
                        className="glass group relative overflow-hidden rounded-2xl p-5">
              <div className="absolute -right-8 -top-8 h-24 w-24 rounded-full bg-blood-500/0 blur-2xl transition duration-500 group-hover:bg-blood-500/25" />
              <div className="flex items-center gap-3">
                <span className="grid h-9 w-9 place-items-center rounded-xl bg-blood-500/15 text-blood-300"><st.icon className="h-4.5 w-4.5" /></span>
                <span className="font-mono text-xs text-fg-3">0{i + 1}</span>
              </div>
              <h3 className="mt-4 font-display text-lg font-semibold">{st.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-fg-3">{st.body}</p>
              {st.key && overview && (
                <p className="mt-3 truncate font-mono text-[11px] text-fg-3/80" title={overview.models[st.key]}>{overview.models[st.key]}</p>
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
