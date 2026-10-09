import { FlaskConical, Quote } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, LabelList, Line, LineChart, ReferenceArea, ResponsiveContainer,
  Tooltip as RTooltip, XAxis, YAxis,
} from "recharts";
import { Card, NotReady, Pill, Section, Spinner, Tooltip } from "../components/ui";
import { api, type Themes as ThemesData } from "../lib/api";
import { useAsync } from "../lib/hooks";
import { SERIES, diverging } from "../lib/palette";

const BAR = "#e66767";
// Short labels for the arc bands; arcs too narrow to label rely on the tooltip.
const ARC_SHORT: Record<string, string> = {
  "Land of Waves": "Waves", "Chūnin Exams": "Chūnin Exams", "Konoha Crush": "Crush",
  "Search for Tsunade": "Tsunade", "Sasuke Recovery Mission": "Sasuke Retrieval", "Post-Recovery (filler)": "Filler arcs",
};
const MAX_TRACES = 4;

function rolling(values: number[], window: number) {
  const half = Math.floor(window / 2);
  return values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - half), i + half + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

function ProfileBars({ profile }: { profile: ThemesData["profile"] }) {
  return (
    <ResponsiveContainer width="100%" height={Math.max(260, profile.length * 34)}>
      <BarChart data={profile} layout="vertical" margin={{ left: 8, right: 48, top: 4, bottom: 4 }} barCategoryGap={6}>
        <CartesianGrid horizontal={false} />
        <XAxis type="number" tickLine={false} axisLine={false} tickFormatter={(v) => v.toFixed(2)} />
        <YAxis type="category" dataKey="theme" width={118} tickLine={false} axisLine={false} />
        <RTooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} content={({ active, payload }) =>
          active && payload?.[0] ? (
            <Tooltip>
              <p className="font-medium text-fg">{payload[0].payload.theme}</p>
              <p className="text-fg-2">mean P(entailment) {Number(payload[0].value).toFixed(3)}</p>
            </Tooltip>
          ) : null} />
        <Bar dataKey="score" fill={BAR} radius={[0, 4, 4, 0]} maxBarSize={22} animationDuration={900}>
          <LabelList dataKey="score" position="right" formatter={(v) => Number(v).toFixed(2)} className="fill-fg-2 text-xs" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function ArcHeatmap({ data }: { data: ThemesData }) {
  const all = data.arcs.flatMap((a) => Object.values(a.lift));
  const span = Math.max(...all.map((v) => Math.abs(Math.log2(v))), 0.05);
  return (
    <div className="overflow-x-auto">
      <div className="grid min-w-[640px] gap-[2px]" style={{ gridTemplateColumns: `150px repeat(${data.labels.length}, minmax(0, 1fr))` }}>
        <div />
        {data.labels.map((l) => (
          <div key={l} className="relative h-20">
            <span className="absolute bottom-2 left-1/2 origin-bottom-left -rotate-45 whitespace-nowrap text-[11px] text-fg-3">{l}</span>
          </div>
        ))}
        {data.arcs.map((row) => (
          <div key={row.arc} className="contents">
            <div className="flex items-center pr-2 text-xs text-fg-2">{row.arc}</div>
            {data.labels.map((l) => {
              const v = row.lift[l];
              return (
                <div key={l} title={`${row.arc} · ${l}: ${v.toFixed(2)}× the series average`}
                     className="grid h-10 place-items-center rounded-[4px] text-[11px] tabular-nums text-white/90 transition hover:ring-2 hover:ring-white/40"
                     style={{ background: diverging(Math.log2(v) / span) }}>
                  {v.toFixed(2)}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="mt-4 flex items-center gap-3 text-xs text-fg-3">
        <span>under-indexed</span>
        <div className="h-2 w-40 rounded-full" style={{ background: `linear-gradient(90deg, ${diverging(-1)}, ${diverging(0)}, ${diverging(1)})` }} />
        <span>over-indexed</span>
        <span className="ml-2 text-fg-3/70">1.00 = series average</span>
      </div>
    </div>
  );
}

function Trajectory({ data }: { data: ThemesData }) {
  const [slots, setSlots] = useState<Record<string, number>>(() =>
    Object.fromEntries(data.profile.slice(0, 3).map((p, i) => [p.theme, i])));
  const chosen = Object.keys(slots);
  const toggle = (theme: string) => setSlots((s) => {
    if (theme in s) { const { [theme]: _, ...rest } = s; return rest; }
    if (Object.keys(s).length >= MAX_TRACES) return s;
    const used = new Set(Object.values(s));
    return { ...s, [theme]: [0, 1, 2, 3].find((i) => !used.has(i))! };
  });
  const rows = useMemo(() => {
    const eps = [...data.episodes].sort((a, b) => a.episode - b.episode);
    const smooth = Object.fromEntries(data.labels.map((l) => [l, rolling(eps.map((e) => Number(e[l])), 9)]));
    return eps.map((e, i) => ({ episode: e.episode, arc: e.arc, ...Object.fromEntries(data.labels.map((l) => [l, smooth[l][i]])) }));
  }, [data]);

  return (
    <>
      <div className="mb-5 flex flex-wrap gap-2">
        {data.profile.map(({ theme }) => (
          <Pill key={theme} active={theme in slots} color={theme in slots ? SERIES[slots[theme]] : undefined}
                onClick={() => toggle(theme)} disabled={!(theme in slots) && chosen.length >= MAX_TRACES}>
            {theme}
          </Pill>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={rows} margin={{ left: 0, right: 12, top: 8, bottom: 0 }}>
          {data.arcBounds.map((a, i) => (
            <ReferenceArea key={a.arc} x1={a.first} x2={a.last} fill={i % 2 ? "#ffffff" : "#e5383b"} fillOpacity={i % 2 ? 0.015 : 0.035}
                           label={ARC_SHORT[a.arc] ? { value: ARC_SHORT[a.arc], position: "insideTop", fill: "#8a8984", fontSize: 10 } : undefined} />
          ))}
          <CartesianGrid vertical={false} />
          <XAxis dataKey="episode" type="number" domain={[1, 220]} ticks={[1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]} tickLine={false} />
          <YAxis tickLine={false} axisLine={false} width={44} tickFormatter={(v) => v.toFixed(2)} />
          <RTooltip content={({ active, payload, label }) => active && payload?.length ? (
            <Tooltip>
              <p className="mb-1 font-medium text-fg">Episode {label} · {payload[0].payload.arc}</p>
              {payload.map((p) => (
                <p key={String(p.dataKey)} className="flex items-center gap-2 text-fg-2">
                  <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />{String(p.dataKey)}
                  <span className="ml-auto pl-4 tabular-nums text-fg">{Number(p.value).toFixed(3)}</span>
                </p>
              ))}
            </Tooltip>) : null} />
          {chosen.map((t) => (
            <Line key={t} dataKey={t} stroke={SERIES[slots[t]]} strokeWidth={2} dot={false} activeDot={{ r: 4, strokeWidth: 2, stroke: "#121215" }} animationDuration={800} />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-fg-3">9-episode rolling mean. Shaded bands are story arcs; hover for the exact arc of any episode.</p>
    </>
  );
}

function Excerpts({ labels }: { labels: string[] }) {
  const [theme, setTheme] = useState(labels[0]);
  const excerpts = useAsync(() => api.excerpts(theme), [theme]);
  return (
    <>
      <div className="mb-5 flex flex-wrap gap-2">
        {labels.map((l) => <Pill key={l} active={l === theme} onClick={() => setTheme(l)}>{l}</Pill>)}
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <AnimatePresence mode="popLayout">
          {(excerpts.data ?? []).map((e, i) => (
            <motion.figure key={`${theme}-${e.episode}-${i}`} layout initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                           exit={{ opacity: 0 }} transition={{ delay: i * 0.06 }}
                           className="flex flex-col rounded-xl border border-line bg-surface-2/60 p-4">
              <Quote className="h-4 w-4 text-blood-400" />
              <blockquote className="mt-2 line-clamp-[9] flex-1 text-sm leading-relaxed text-fg-2">{e.text}</blockquote>
              <figcaption className="mt-3 flex items-center justify-between text-xs text-fg-3">
                <span>Ep {e.episode} · {e.arc}</span>
                <span className="rounded-full bg-white/5 px-2 py-0.5 tabular-nums text-fg-2">{e.score.toFixed(2)}</span>
              </figcaption>
            </motion.figure>
          ))}
        </AnimatePresence>
      </div>
    </>
  );
}

function ThemeLab() {
  const [labels, setLabels] = useState("courage, rivalry, redemption");
  const [range, setRange] = useState({ first: 1, last: 5 });
  const [state, setState] = useState<{ loading?: boolean; error?: string; result?: Awaited<ReturnType<typeof api.scoreThemes>> }>({});
  const run = async () => {
    setState({ loading: true });
    try {
      const list = labels.split(",").map((s) => s.trim()).filter(Boolean);
      setState({ result: await api.scoreThemes(list, range.first, range.last) });
    } catch (e) { setState({ error: (e as Error).message }); }
  };
  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
      <div className="space-y-4">
        <label className="block text-sm text-fg-2">Themes (comma separated)
          <input value={labels} onChange={(e) => setLabels(e.target.value)}
                 className="mt-1.5 w-full rounded-xl border border-line bg-surface-2 px-4 py-2.5 text-fg outline-none transition focus:border-blood-500/60 focus:ring-2 focus:ring-blood-500/20" />
        </label>
        <div className="grid grid-cols-2 gap-3">
          {(["first", "last"] as const).map((k) => (
            <label key={k} className="text-sm text-fg-2">{k === "first" ? "From episode" : "To episode"}
              <input type="number" min={1} max={220} value={range[k]} onChange={(e) => setRange({ ...range, [k]: Number(e.target.value) })}
                     className="mt-1.5 w-full rounded-xl border border-line bg-surface-2 px-4 py-2.5 text-fg outline-none focus:border-blood-500/60" />
            </label>
          ))}
        </div>
        <button onClick={run} disabled={state.loading}
                className="inline-flex items-center gap-2 rounded-full bg-blood-500 px-5 py-2.5 font-medium text-white transition hover:bg-blood-400 disabled:opacity-60">
          <FlaskConical className="h-4 w-4" /> {state.loading ? "Scoring…" : "Score live"}
        </button>
        <p className="text-xs text-fg-3">Runs the NLI model on demand (up to 20 episodes). On CPU expect a few seconds per episode.</p>
        {state.error && <p className="text-sm text-blood-300">{state.error}</p>}
      </div>
      <div className="min-h-[220px]">
        {state.loading && <Spinner label="Reading the episodes" />}
        {state.result && <ProfileBars profile={state.result.profile} />}
      </div>
    </div>
  );
}

export function Themes() {
  const themes = useAsync(api.themes, []);
  return (
    <Section id="themes" eyebrow="01 · Zero-shot classification" title={<>What the series is <span className="text-gradient-blood">about</span></>}
             lead={<>Each episode is split into token-budgeted chunks. A natural-language-inference model scores every chunk
               against hypotheses like “This dialogue is about sacrifice.” Opening and ending song lyrics, repeated
               across up to 78 episodes, are removed first so they can't inflate themes like <em>hope</em>.</>}>
      {themes.loading && !themes.data ? <Spinner /> : !themes.data ? <NotReady error={themes.error} command="sharingan themes" /> : (
        <div className="grid gap-6">
          <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
            <Card title="Series theme profile" subtitle={`Mean entailment probability across all chunks · ${themes.data.model.split("/").pop()}`}>
              <ProfileBars profile={themes.data.profile} />
            </Card>
            <Card title="Themes by story arc" subtitle="Arc mean ÷ series mean: red runs hotter than usual, blue cooler">
              <ArcHeatmap data={themes.data} />
            </Card>
          </div>
          <Card title="Theme intensity across 220 episodes" subtitle="Choose up to four themes to trace">
            <Trajectory data={themes.data} />
          </Card>
          <Card title="Most representative dialogue" subtitle="The highest-scoring chunk of dialogue for each theme">
            <Excerpts labels={themes.data.profile.map((p) => p.theme)} />
          </Card>
          <Card title="Theme lab" subtitle="Write your own hypotheses and score them on any episode range">
            <ThemeLab />
          </Card>
        </div>
      )}
    </Section>
  );
}
