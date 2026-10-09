import { forceCollide, forceX, forceY } from "d3-force";
import { Crosshair, Users } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods, type LinkObject, type NodeObject } from "react-force-graph-2d";
import { Card, NotReady, Pill, Section, Spinner } from "../components/ui";
import { api, type CharacterNode, type Network as NetworkData } from "../lib/api";
import { fmt, useAsync, useElementSize } from "../lib/hooks";
import { OTHER, SERIES, seriesColor } from "../lib/palette";

type GNode = NodeObject<CharacterNode>;
type GLink = LinkObject<GNode, { weight: number; episodes: number; cooccurrences: number }>;
const ALL = "all";
const endpoint = (v: unknown) => (typeof v === "object" && v ? (v as GNode).id : v) as string;

function Graph({ data, selected, onSelect }: { data: NetworkData; selected?: string; onSelect: (id?: string) => void }) {
  const [box, size] = useElementSize<HTMLDivElement>();
  const fg = useRef<ForceGraphMethods<GNode, GLink> | undefined>(undefined);
  const [hover, setHover] = useState<string>();

  const graph = useMemo(() => ({
    // Ascending PageRank: the most central characters are painted last, on top.
    nodes: [...data.nodes].sort((a, b) => a.pagerank - b.pagerank).map((n) => ({ ...n })) as GNode[],
    links: data.links.map((l) => ({ ...l })) as GLink[],
  }), [data]);
  const maxPr = useMemo(() => Math.max(...data.nodes.map((n) => n.pagerank)), [data]);
  const maxW = useMemo(() => Math.max(...data.links.map((l) => l.weight)), [data]);
  const neighbours = useMemo(() => {
    const m = new Map<string, Set<string>>();
    data.links.forEach((l) => {
      (m.get(l.source) ?? m.set(l.source, new Set()).get(l.source)!).add(l.target);
      (m.get(l.target) ?? m.set(l.target, new Set()).get(l.target)!).add(l.source);
    });
    return m;
  }, [data]);
  const labelled = useMemo(() => new Set([...data.nodes].sort((a, b) => b.pagerank - a.pagerank).slice(0, 22).map((n) => n.id)), [data]);
  const focus = hover ?? selected;
  const isLit = (id: string) => !focus || id === focus || neighbours.get(focus)?.has(id);
  const radius = (n: CharacterNode) => 3 + 17 * Math.sqrt(n.pagerank / maxPr);

  useEffect(() => {
    const f = fg.current;
    if (!f) return;
    f.d3Force("charge")?.strength(-420).distanceMax(600);
    f.d3Force("x", forceX(0).strength(0.04));
    f.d3Force("y", forceY(0).strength(0.04));
    f.d3Force("collide", forceCollide<GNode>((n) => radius(n as CharacterNode) * 1.35 + 8));
    (f.d3Force("link") as unknown as { distance: (fn: (l: GLink) => number) => void } | undefined)
      ?.distance((l: GLink) => 60 + 160 * (1 - Math.sqrt(l.weight / maxW)));
    f.d3ReheatSimulation();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph, maxW]);

  return (
    <div ref={box} className="relative h-[560px] overflow-hidden rounded-xl lg:h-[760px] bg-[radial-gradient(ellipse_at_center,#1a1214_0%,#0d0d0f_70%)] sm:h-[640px]">
      {size.width > 0 && (
        <ForceGraph2D<CharacterNode, GLink>
          ref={fg} graphData={graph} width={size.width} height={size.height} backgroundColor="rgba(0,0,0,0)"
          cooldownTicks={220} d3VelocityDecay={0.3} warmupTicks={60} onEngineStop={() => fg.current?.zoomToFit(700, 60)}
          nodeRelSize={1} nodeVal={(n) => radius(n) ** 2}
          nodeLabel={() => ""}
          onNodeHover={(n) => setHover(n?.id as string | undefined)}
          onNodeClick={(n) => { onSelect(n.id as string); fg.current?.centerAt(n.x, n.y, 600); }}
          onBackgroundClick={() => onSelect(undefined)}
          linkColor={(l) => {
            const lit = focus && (endpoint(l.source) === focus || endpoint(l.target) === focus);
            return lit ? "rgba(255,138,143,0.75)" : focus ? "rgba(195,194,183,0.04)" : "rgba(195,194,183,0.16)";
          }}
          linkWidth={(l) => 0.4 + 5 * (l.weight / maxW) ** 0.6}
          linkDirectionalParticles={(l) => (focus && (endpoint(l.source) === focus || endpoint(l.target) === focus) ? 2 : 0)}
          linkDirectionalParticleWidth={2.2} linkDirectionalParticleColor={() => "#ff8a8f"}
          nodeCanvasObject={(node, ctx, scale) => {
            const n = node as GNode & CharacterNode;
            const r = radius(n), lit = isLit(n.id as string), color = seriesColor(n.community);
            ctx.globalAlpha = lit ? 1 : 0.12;
            ctx.shadowColor = color; ctx.shadowBlur = lit ? 18 : 0;
            ctx.beginPath(); ctx.arc(n.x!, n.y!, r, 0, 2 * Math.PI); ctx.fillStyle = color; ctx.fill();
            ctx.shadowBlur = 0;
            ctx.lineWidth = n.id === focus ? 2.5 / scale + 1 : 1.5 / scale; ctx.strokeStyle = n.id === focus ? "#fff" : "#0a0a0c"; ctx.stroke();
            ctx.globalAlpha = 1;
          }}
          onRenderFramePost={(ctx, scale) => {
            // Second pass so no node is painted over another node's label.
            for (const n of graph.nodes) {
              const name = String(n.id);
              if (!isLit(name) || !(labelled.has(name) || focus)) continue;
              const r = radius(n);
              ctx.textAlign = "center";
              ctx.font = `600 ${Math.min(r * 0.5, 16)}px Inter, sans-serif`;
              const inside = r * scale > 26 && ctx.measureText(name).width < 1.8 * r;
              if (!inside) ctx.font = `600 ${11 / scale}px Inter, sans-serif`;
              const y = inside ? n.y! : n.y! + r + 3 / scale;
              ctx.textBaseline = inside ? "middle" : "top";
              ctx.lineWidth = 3 / scale; ctx.strokeStyle = "rgba(10,10,12,0.85)"; ctx.strokeText(name, n.x!, y);
              ctx.fillStyle = "#fff"; ctx.fillText(name, n.x!, y);
            }
          }}
          nodePointerAreaPaint={(node, color, ctx) => {
            ctx.fillStyle = color; ctx.beginPath(); ctx.arc(node.x!, node.y!, radius(node as CharacterNode) + 4, 0, 2 * Math.PI); ctx.fill();
          }}
        />
      )}
      <div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2 rounded-full bg-ink/70 px-3 py-1 text-xs text-fg-3 backdrop-blur">
        <Crosshair className="h-3.5 w-3.5" /> hover to focus · click for details · scroll to zoom
      </div>
    </div>
  );
}

function Profile({ node, arc }: { node: CharacterNode; arc: string }) {
  const ties = useAsync(() => api.ties(node.id, arc), [node.id, arc]);
  const max = Math.max(...(ties.data ?? []).map((t) => t.weight), 1);
  const stats = [
    ["Mentions", fmt.format(node.mentions)], ["Connections", node.degree],
    ["PageRank", node.pagerank.toFixed(3)], ["Betweenness", node.betweenness.toFixed(3)],
  ];
  return (
    <motion.div key={node.id} initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}>
      <div className="flex items-center gap-3">
        <span className="h-10 w-10 rounded-full shadow-[0_0_20px_currentColor]" style={{ background: seriesColor(node.community), color: seriesColor(node.community) }} />
        <div>
          <h4 className="font-display text-xl font-semibold">{node.id}</h4>
          <p className="text-xs text-fg-3">Community {node.community + 1}</p>
        </div>
      </div>
      <dl className="mt-5 grid grid-cols-2 gap-2">
        {stats.map(([k, v]) => (
          <div key={k} className="rounded-xl bg-surface-2/70 px-3 py-2.5">
            <dt className="text-[11px] uppercase tracking-wider text-fg-3">{k}</dt>
            <dd className="mt-0.5 font-display text-lg tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mb-2 mt-6 text-xs uppercase tracking-wider text-fg-3">Strongest ties</p>
      <ul className="space-y-2">
        {(ties.data ?? []).map((t) => (
          <li key={t.character} className="text-sm">
            <div className="flex justify-between text-fg-2"><span>{t.character}</span><span className="tabular-nums text-fg-3">{t.episodes} ep</span></div>
            <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-white/5">
              <motion.div initial={{ width: 0 }} animate={{ width: `${(100 * t.weight) / max}%` }} transition={{ duration: 0.7 }}
                          className="h-full rounded-full bg-gradient-to-r from-blood-600 to-blood-300" />
            </div>
          </li>
        ))}
      </ul>
    </motion.div>
  );
}

export function Network() {
  const [arc, setArc] = useState(ALL);
  const [edges, setEdges] = useState(250);
  const [selected, setSelected] = useState<string>();
  const net = useAsync(() => api.network(arc, edges), [arc, edges]);
  const data = net.data;
  const node = data?.nodes.find((n) => n.id === selected) ?? data?.nodes.find((n) => n.id === data.ranking[0]?.character);
  const maxPr = data ? Math.max(...data.ranking.map((r) => r.pagerank)) : 1;
  const communityNames = useMemo(() => {
    const byCommunity = new Map<number, string[]>();
    [...(data?.nodes ?? [])].sort((a, b) => b.pagerank - a.pagerank).forEach((n) => {
      const list = byCommunity.get(n.community) ?? byCommunity.set(n.community, []).get(n.community)!;
      if (list.length < 2) list.push(n.id);
    });
    return [...byCommunity.entries()].sort((a, b) => a[0] - b[0]);
  }, [data]);

  return (
    <Section id="network" eyebrow="02 · Named entities + graph theory" title={<>Who matters to <span className="text-gradient-blood">whom</span></>}
             lead={<>spaCy's transformer NER, backed by a cast gazetteer and alias resolution (“Pervy Sage” → Jiraiya,
               “Sasuke-kun” → Sasuke), finds every character mention. Two characters are linked when they're named
               within ten subtitle lines of each other, weighted by <code className="font-mono text-sm">exp(−distance/τ)</code>.
               Communities come from Louvain modularity.</>}>
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <Pill active={arc === ALL} onClick={() => setArc(ALL)}>Whole series</Pill>
        {(data?.arcs ?? []).map((a) => <Pill key={a} active={arc === a} onClick={() => { setArc(a); setSelected(undefined); }}>{a}</Pill>)}
      </div>
      {!data && net.loading ? <Spinner label="Building the network" /> : !data ? <NotReady error={net.error} command="sharingan characters" /> : (
        <div className="grid items-start gap-6 lg:grid-cols-[1fr_320px]">
          <Card className="!p-3">
            <Graph data={data} selected={selected} onSelect={setSelected} />
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-3 pb-2 pt-4 text-xs text-fg-3">
              <label className="flex items-center gap-3">Relationships drawn
                <input type="range" min={50} max={600} step={25} value={edges} onChange={(e) => setEdges(Number(e.target.value))} className="accent-blood-500" />
                <span className="w-8 tabular-nums text-fg-2">{edges}</span>
              </label>
              <span className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
                {communityNames.filter(([c]) => c < SERIES.length).map(([c, names]) => (
                  <span key={c} className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full" style={{ background: SERIES[c] }} />{names.join(" · ")}</span>
                ))}
                {communityNames.some(([c]) => c >= SERIES.length) && <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full" style={{ background: OTHER }} />smaller groups</span>}
              </span>
            </div>
          </Card>
          <div className="grid gap-6">
            <Card><AnimatePresence mode="wait">{node && <Profile node={node} arc={arc} />}</AnimatePresence></Card>
            <Card title={<span className="flex items-center gap-2"><Users className="h-4 w-4 text-blood-400" />Centrality ranking</span>}>
              <ol className="space-y-1.5">
                {data.ranking.slice(0, 10).map((r, i) => (
                  <li key={r.character}>
                    <button onClick={() => setSelected(r.character)} className="group flex w-full items-center gap-3 rounded-lg px-1.5 py-1 text-left text-sm hover:bg-white/5">
                      <span className="w-4 text-right font-mono text-xs text-fg-3">{i + 1}</span>
                      <span className="h-2 w-2 rounded-full" style={{ background: seriesColor(r.community) }} />
                      <span className="flex-1 text-fg-2 group-hover:text-fg">{r.character}</span>
                      <span className="h-1 w-16 overflow-hidden rounded-full bg-white/5"><span className="block h-full rounded-full bg-fg-3" style={{ width: `${(100 * r.pagerank) / maxPr}%` }} /></span>
                    </button>
                  </li>
                ))}
              </ol>
            </Card>
          </div>
        </div>
      )}
    </Section>
  );
}
