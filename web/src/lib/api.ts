// Typed client for the Sharingan FastAPI backend.

export type Overview = {
  version: string; episodes: number; lines: number; jutsus: number; characters: number;
  relationships: number; personas: number;
  models: Record<"themes" | "ner" | "jutsu" | "persona" | "embeddings", string>;
  ready: Record<"themes" | "characters" | "jutsu" | "persona", boolean>;
};

export type ThemeRow = { episode: number; arc: string } & Record<string, number | string>;
export type Themes = {
  labels: string[]; model: string;
  profile: { theme: string; score: number; relative: number }[];
  episodes: ThemeRow[];
  arcs: { arc: string; lift: Record<string, number> }[];
  arcBounds: { arc: string; first: number; last: number }[];
};
export type Excerpt = { episode: number; arc: string; score: number; text: string };

export type CharacterNode = {
  id: string; mentions: number; degree: number; strength: number; pagerank: number;
  betweenness: number; eigenvector: number; community: number;
};
export type CharacterLink = { source: string; target: string; weight: number; cooccurrences: number; episodes: number };
export type Network = {
  arc: string; arcs: string[]; nodes: CharacterNode[]; links: CharacterLink[];
  ranking: (CharacterNode & { character: string })[]; communities: number;
};
export type Tie = { character: string; weight: number; episodes: number };

export type ClassScores = { precision: number; recall: number; "f1-score": number; support: number };
export type ModelReport = {
  accuracy: number; macro_f1: number; per_class: Record<string, ClassScores>;
  confusion_matrix: { labels: string[]; matrix: number[][] }; train_seconds: number; model?: string;
};
export type Stat = { mean: number; std: number };
export type CvModel = { macro_f1: Stat; accuracy: Stat; per_class_f1: Record<string, Stat>; folds: number[]; model?: string };
export type CrossVal = {
  folds: number; n: number; models: Record<string, CvModel>;
  transformer_minus_tfidf?: { mean: number; std: number; folds_won: number; per_fold: number[] };
};
export type JutsuReport = {
  data: Record<string, number>; label_counts: Record<string, number>;
  baselines: Record<string, ModelReport>; transformer: ModelReport; crossval: CrossVal | null;
};
export type Prediction = { label: string; probabilities: Record<string, number> };

export type Persona = { name: string; fullName: string; identity: string; greeting: string; speech: string; hasAdapter: boolean };
export type ChatTurn = { role: "user" | "assistant"; content: string };
export type Retrieved = { voice: string[]; scenes: string[] };

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  overview: () => request<Overview>("/api/overview"),
  themes: () => request<Themes>("/api/themes"),
  excerpts: (theme: string) => request<Excerpt[]>(`/api/themes/excerpts?theme=${encodeURIComponent(theme)}&k=3`),
  scoreThemes: (labels: string[], first: number, last: number) =>
    request<Pick<Themes, "labels" | "profile" | "episodes">>("/api/themes/score", {
      method: "POST", body: JSON.stringify({ labels, first, last }),
    }),
  network: (arc: string, topEdges: number) =>
    request<Network>(`/api/characters?arc=${encodeURIComponent(arc)}&top_edges=${topEdges}`),
  ties: (name: string, arc: string) =>
    request<Tie[]>(`/api/characters/ties?name=${encodeURIComponent(name)}&arc=${encodeURIComponent(arc)}`),
  classify: (text: string) => request<Prediction>("/api/jutsu/classify", { method: "POST", body: JSON.stringify({ text }) }),
  jutsuReport: () => request<JutsuReport>("/api/jutsu/report"),
  personas: () => request<Persona[]>("/api/personas"),
};

/** POST /api/chat and consume its server-sent events. */
export async function streamChat(
  body: { character: string; message: string; history: ChatTurn[] },
  on: { context?: (r: Retrieved) => void; delta: (t: string) => void; error?: (m: string) => void },
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal,
  });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new ApiError(res.status, detail);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let split: number;
    while ((split = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const event = /^event: (.*)$/m.exec(raw)?.[1] ?? "message";
      const data = JSON.parse(/^data: (.*)$/m.exec(raw)?.[1] ?? "null");
      if (event === "context") on.context?.(data);
      else if (event === "delta") on.delta(data);
      else if (event === "error") on.error?.(data);
      else if (event === "done") return;
    }
  }
}
