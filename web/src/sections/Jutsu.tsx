import { Eye, Flame, Hand, Wand2 } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { Card, NotReady, Section, Spinner } from "../components/ui";
import { api, type CrossVal, type JutsuReport, type ModelReport, type Prediction } from "../lib/api";
import { fmt, useAsync } from "../lib/hooks";
import { SERIES, sequential } from "../lib/palette";

const CLASSES = ["Genjutsu", "Ninjutsu", "Taijutsu"];
const META: Record<string, { icon: typeof Eye; blurb: string }> = {
  Genjutsu: { icon: Eye, blurb: "Illusion: hijacks the target's senses through their chakra." },
  Ninjutsu: { icon: Flame, blurb: "Chakra-moulded techniques: elements, clones, sealing, summoning." },
  Taijutsu: { icon: Hand, blurb: "Pure physical combat: speed, strength and martial arts." },
};
const colorOf = (label: string) => SERIES[CLASSES.indexOf(label)] ?? SERIES[0];
const EXAMPLES = [
  "The user traps the target in an illusion where they relive their worst memory for what feels like days, while only a moment passes in reality.",
  "A relentless flurry of kicks launches the opponent into the air, followed by a spinning drop that slams them into the ground.",
  "The user moulds wind-natured chakra into a rapidly rotating sphere and drives it into the enemy's body.",
];
const NAMES: Record<string, string> = { tfidf_logreg: "TF-IDF + LogReg", embedding_logreg: "MiniLM embeddings + LogReg" };

function Classifier() {
  const [text, setText] = useState(EXAMPLES[0]);
  const [state, setState] = useState<{ loading?: boolean; error?: string; result?: Prediction }>({});
  const run = async () => {
    setState({ loading: true });
    try { setState({ result: await api.classify(text) }); } catch (e) { setState({ error: (e as Error).message }); }
  };
  const top = state.result?.label;
  const Icon = top ? META[top].icon : Wand2;
  return (
    <Card title="Classify a technique" subtitle="Describe any jutsu, real or invented">
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={5}
                className="w-full resize-none rounded-xl border border-line bg-surface-2 px-4 py-3 text-sm leading-relaxed text-fg outline-none transition focus:border-blood-500/60 focus:ring-2 focus:ring-blood-500/20" />
      <div className="mt-3 flex flex-wrap gap-2">
        {EXAMPLES.map((ex, i) => (
          <button key={i} onClick={() => setText(ex)} className="rounded-full border border-line px-3 py-1 text-xs text-fg-3 transition hover:border-white/20 hover:text-fg">
            Example {i + 1}
          </button>
        ))}
        <button onClick={run} disabled={state.loading || text.trim().length < 3}
                className="ml-auto inline-flex items-center gap-2 rounded-full bg-blood-500 px-5 py-2 text-sm font-medium text-white transition hover:bg-blood-400 disabled:opacity-60">
          <Wand2 className="h-4 w-4" /> {state.loading ? "Reading…" : "Classify"}
        </button>
      </div>
      {state.error && <p className="mt-4 text-sm text-blood-300">{state.error}</p>}
      <AnimatePresence>
        {state.result && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-6 rounded-xl border border-line bg-surface-2/50 p-5">
            <div className="flex items-center gap-4">
              <span className="grid h-12 w-12 place-items-center rounded-2xl" style={{ background: `${colorOf(top!)}26`, color: colorOf(top!) }}>
                <Icon className="h-6 w-6" />
              </span>
              <div>
                <p className="font-display text-2xl font-semibold">{top}</p>
                <p className="text-sm text-fg-3">{META[top!].blurb}</p>
              </div>
            </div>
            <div className="mt-5 space-y-3">
              {Object.entries(state.result.probabilities).map(([label, p]) => (
                <div key={label}>
                  <div className="mb-1 flex justify-between text-sm"><span className="text-fg-2">{label}</span><span className="tabular-nums text-fg">{(100 * p).toFixed(1)}%</span></div>
                  <div className="h-2 overflow-hidden rounded-full bg-white/5">
                    <motion.div initial={{ width: 0 }} animate={{ width: `${100 * p}%` }} transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                                className="h-full rounded-full" style={{ background: colorOf(label) }} />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}

function Leaderboard({ report }: { report: JutsuReport }) {
  const rows: [string, ModelReport, boolean][] = [
    ...Object.entries(report.baselines).map(([k, v]) => [NAMES[k] ?? k, v, false] as [string, ModelReport, boolean]),
    [`Fine-tuned ${report.transformer.model?.split("/").pop()}`, report.transformer, true],
  ];
  const best = Math.max(...rows.map(([, r]) => r.macro_f1));
  return (
    <Card title="Held-out test split" subtitle={`${report.data.test} unseen jutsu (only ${report.transformer.per_class.Genjutsu?.support ?? "a few"} Genjutsu) · model selected on validation macro-F1`}>
      <div className="space-y-4">
        {rows.map(([name, r, ours]) => (
          <div key={name}>
            <div className="mb-1.5 flex items-baseline justify-between gap-3">
              <span className={ours ? "font-medium text-fg" : "text-fg-2"}>{name}{r.macro_f1 === best && <span className="ml-2 rounded-full bg-blood-500/15 px-2 py-0.5 text-[10px] uppercase tracking-wider text-blood-300">best</span>}</span>
              <span className="text-xs tabular-nums text-fg-3">acc {(100 * r.accuracy).toFixed(1)}% · {r.train_seconds}s</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-white/5">
                <motion.div initial={{ width: 0 }} whileInView={{ width: `${100 * r.macro_f1}%` }} viewport={{ once: true }} transition={{ duration: 0.9 }}
                            className={ours ? "h-full rounded-full bg-gradient-to-r from-blood-600 to-blood-300" : "h-full rounded-full bg-fg-3/60"} />
              </div>
              <span className="w-14 text-right font-display tabular-nums">{r.macro_f1.toFixed(3)}</span>
            </div>
            <div className="mt-1.5 flex gap-4 text-[11px] text-fg-3">
              {CLASSES.map((c) => <span key={c}>{c} F1 <span className="tabular-nums text-fg-2">{r.per_class[c]?.["f1-score"].toFixed(2)}</span></span>)}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/** k-fold results: dot = mean, bar = ±1 std, ticks = individual folds (shared 0.6–1.0 axis). */
function CrossValidation({ cv, transformerName }: { cv: CrossVal; transformerName?: string }) {
  const lo = 0.6, hi = 1.0, pos = (v: number) => `${(100 * (Math.min(hi, Math.max(lo, v)) - lo)) / (hi - lo)}%`;
  const label = (k: string) => (k === "transformer" ? `Fine-tuned ${transformerName ?? "transformer"}` : NAMES[k] ?? k);
  const d = cv.transformer_minus_tfidf;
  return (
    <Card title={`${cv.folds}-fold cross-validation`} subtitle={`All ${fmt.format(cv.n)} jutsu · identical folds for every model · macro-F1`}>
      <div className="space-y-5">
        {Object.entries(cv.models).map(([k, m]) => (
          <div key={k}>
            <div className="mb-2 flex items-baseline justify-between text-sm">
              <span className={k === "transformer" ? "font-medium text-fg" : "text-fg-2"}>{label(k)}</span>
              <span className="font-display tabular-nums">{m.macro_f1.mean.toFixed(3)} <span className="text-xs text-fg-3">± {m.macro_f1.std.toFixed(3)}</span></span>
            </div>
            <div className="relative h-6">
              <div className="absolute inset-x-0 top-1/2 h-px bg-white/10" />
              <div className="absolute top-1/2 h-2 -translate-y-1/2 rounded-full bg-white/10"
                   style={{ left: pos(m.macro_f1.mean - m.macro_f1.std), width: `calc(${pos(m.macro_f1.mean + m.macro_f1.std)} - ${pos(m.macro_f1.mean - m.macro_f1.std)})` }} />
              {m.folds.map((f, i) => (
                <div key={i} title={`fold ${i + 1}: ${f.toFixed(3)}`} className="absolute top-1/2 h-3 w-px -translate-y-1/2 bg-fg-3" style={{ left: pos(f) }} />
              ))}
              <div className={`absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-surface ${k === "transformer" ? "bg-blood-400" : "bg-fg-2"}`}
                   style={{ left: pos(m.macro_f1.mean) }} />
            </div>
          </div>
        ))}
        <div className="flex justify-between text-[11px] tabular-nums text-fg-3"><span>0.60</span><span>0.70</span><span>0.80</span><span>0.90</span><span>1.00</span></div>
      </div>
      {d && (
        <p className="mt-5 rounded-xl bg-surface-2/70 px-4 py-3 text-sm text-fg-2">
          Paired difference (fine-tuned − TF-IDF): <span className="font-display tabular-nums text-fg">{d.mean >= 0 ? "+" : ""}{d.mean.toFixed(3)} ± {d.std.toFixed(3)}</span>,
          better on <span className="text-fg">{d.folds_won} of {cv.folds}</span> folds.
        </p>
      )}
    </Card>
  );
}

function Confusion({ report }: { report: ModelReport }) {
  const { labels, matrix } = report.confusion_matrix;
  return (
    <Card title="Confusion matrix" subtitle="Fine-tuned model · rows are true labels, columns are predictions (row-normalised colour)">
      <div className="grid gap-[2px]" style={{ gridTemplateColumns: `90px repeat(${labels.length}, 1fr)` }}>
        <div />
        {labels.map((l) => <div key={l} className="pb-1 text-center text-xs text-fg-3">{l}</div>)}
        {matrix.map((row, i) => {
          const total = row.reduce((a, b) => a + b, 0) || 1;
          return (
            <div key={labels[i]} className="contents">
              <div className="flex items-center text-xs text-fg-2">{labels[i]}</div>
              {row.map((v, j) => (
                <div key={j} title={`${labels[i]} → ${labels[j]}: ${v}`} className="grid h-14 place-items-center rounded-[4px] font-display text-lg tabular-nums"
                     style={{ background: sequential(v / total), color: v / total > 0.45 ? "#fff" : "#c3c2b7" }}>{v}</div>
              ))}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function Distribution({ counts }: { counts: Record<string, number> }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  return (
    <Card title="Training data" subtitle={`${fmt.format(total)} labelled jutsu from the Naruto Fandom wiki`}>
      <div className="flex h-4 gap-[2px] overflow-hidden rounded-full">
        {CLASSES.map((c) => <div key={c} style={{ width: `${(100 * (counts[c] ?? 0)) / total}%`, background: colorOf(c) }} />)}
      </div>
      <div className="mt-3 flex flex-wrap gap-4 text-sm">
        {CLASSES.map((c) => (
          <span key={c} className="flex items-center gap-2 text-fg-2"><span className="h-2.5 w-2.5 rounded-full" style={{ background: colorOf(c) }} />
            {c} <span className="tabular-nums text-fg-3">{fmt.format(counts[c] ?? 0)} · {((100 * (counts[c] ?? 0)) / total).toFixed(1)}%</span></span>
        ))}
      </div>
      <p className="mt-4 text-xs leading-relaxed text-fg-3">Genjutsu is about 3% of the data, so accuracy alone would reward ignoring it. Training uses
        inverse-frequency class weights, and every model is compared on macro-F1.</p>
    </Card>
  );
}

export function Jutsu() {
  const report = useAsync(api.jutsuReport, []);
  return (
    <Section id="jutsu" eyebrow="03 · Text classification" title={<>Ninjutsu, Genjutsu or <span className="text-gradient-blood">Taijutsu</span>?</>}
             lead="A transformer encoder fine-tuned on scraped jutsu descriptions, measured against two classical baselines on a stratified held-out test split it never saw during training or model selection.">
      <div className="grid gap-6 lg:grid-cols-2">
        <Classifier />
        <div className="grid gap-6">
          {report.loading && !report.data ? <Spinner /> : !report.data ? <Card><NotReady error={report.error} command="sharingan jutsu train" /></Card> : (
            <>
              <Leaderboard report={report.data} />
              {report.data.crossval && <CrossValidation cv={report.data.crossval} transformerName={report.data.transformer.model?.split("/").pop()} />}
              <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
                <Confusion report={report.data.transformer} />
                <Distribution counts={report.data.label_counts} />
              </div>
            </>
          )}
        </div>
      </div>
    </Section>
  );
}
