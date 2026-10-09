import clsx from "clsx";
import { BookMarked, ChevronDown, Film, RotateCcw, SendHorizontal, Sparkles, Square } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { type FormEvent, useEffect, useRef, useState } from "react";
import { NotReady, Section, Spinner } from "../components/ui";
import { api, type ChatTurn, type Persona, type Retrieved, streamChat } from "../lib/api";
import { useAsync } from "../lib/hooks";

type Message = ChatTurn & { retrieved?: Retrieved; error?: boolean };

const AVATAR: Record<string, string> = {
  Naruto: "from-orange-400 to-amber-600", Sasuke: "from-indigo-500 to-slate-800", Sakura: "from-pink-400 to-rose-600",
  Kakashi: "from-slate-300 to-slate-600", Shikamaru: "from-emerald-500 to-teal-800", "Rock Lee": "from-green-500 to-lime-700",
};
const PROMPTS: Record<string, string[]> = {
  Naruto: ["What's your dream?", "What do you think of Sasuke?", "Tell me about Ichiraku ramen."],
  Sasuke: ["Why do you want power?", "What do you think of Naruto?", "Do you trust Kakashi?"],
  Sakura: ["How was training with Lady Tsunade?", "What do you think of Naruto?", "Who's your rival?"],
  Kakashi: ["Why are you always late?", "What's the most important lesson you teach?", "What's in your book?"],
  Shikamaru: ["Want to play shogi?", "How did you fight Temari?", "Why are you so lazy?"],
  "Rock Lee": ["How do you train?", "Who is your rival?", "What does Guy Sensei mean to you?"],
};

function Avatar({ name, size = "md" }: { name: string; size?: "sm" | "md" }) {
  return (
    <span className={clsx("grid shrink-0 place-items-center rounded-full bg-gradient-to-br font-display font-semibold text-white shadow-lg",
                          AVATAR[name] ?? "from-blood-400 to-blood-700", size === "md" ? "h-11 w-11 text-sm" : "h-8 w-8 text-xs")}>
      {name.split(" ").map((w) => w[0]).join("")}
    </span>
  );
}

function Memories({ r }: { r: Retrieved }) {
  const [open, setOpen] = useState(false);
  if (!r.voice.length && !r.scenes.length) return null;
  return (
    <div className="mt-2">
      <button onClick={() => setOpen(!open)} className="inline-flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-fg-3 transition hover:text-fg-2">
        <Sparkles className="h-3 w-3" /> {r.voice.length + r.scenes.length} memories used
        <ChevronDown className={clsx("h-3 w-3 transition", open && "rotate-180")} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="mt-2 space-y-2 rounded-xl border border-line bg-ink/50 p-3 text-xs leading-relaxed">
              {r.voice.map((v, i) => <p key={`v${i}`} className="flex gap-2 text-fg-2"><BookMarked className="mt-0.5 h-3 w-3 shrink-0 text-blood-400" />“{v}”</p>)}
              {r.scenes.map((s, i) => <p key={`s${i}`} className="flex gap-2 text-fg-3"><Film className="mt-0.5 h-3 w-3 shrink-0 text-[#3987e5]" /><span className="line-clamp-3">{s}</span></p>)}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Conversation({ persona }: { persona: Persona }) {
  const [threads, setThreads] = useState<Record<string, Message[]>>({});
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const messages = threads[persona.name] ?? [];

  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" }); }, [messages]);

  const update = (fn: (m: Message[]) => Message[]) => setThreads((t) => ({ ...t, [persona.name]: fn(t[persona.name] ?? []) }));
  const patchLast = (patch: (m: Message) => Message) => update((m) => [...m.slice(0, -1), patch(m[m.length - 1])]);

  const send = async (text: string) => {
    const message = text.trim();
    if (!message || busy) return;
    const history: ChatTurn[] = messages.filter((m) => !m.error).map(({ role, content }) => ({ role, content }));
    update((m) => [...m, { role: "user", content: message }, { role: "assistant", content: "" }]);
    setInput(""); setBusy(true);
    abort.current = new AbortController();
    try {
      await streamChat({ character: persona.name, message, history }, {
        context: (retrieved) => patchLast((m) => ({ ...m, retrieved })),
        delta: (t) => patchLast((m) => ({ ...m, content: m.content + t })),
        error: (e) => patchLast((m) => ({ ...m, content: `Something went wrong: ${e}`, error: true })),
      }, abort.current.signal);
    } catch (e) {
      if ((e as Error).name !== "AbortError") patchLast((m) => ({ ...m, content: (e as Error).message, error: true }));
    } finally { setBusy(false); }
  };
  const submit = (e: FormEvent) => { e.preventDefault(); void send(input); };

  return (
    <div className="glass flex h-[640px] flex-col overflow-hidden rounded-2xl">
      <header className="flex items-center gap-3 border-b border-line px-5 py-4">
        <Avatar name={persona.name} />
        <div className="min-w-0 flex-1">
          <p className="font-display font-semibold">{persona.fullName}</p>
          <p className="truncate text-xs text-fg-3">{persona.speech}</p>
        </div>
        {messages.length > 0 && (
          <button onClick={() => update(() => [])} className="rounded-full p-2 text-fg-3 transition hover:bg-white/5 hover:text-fg" title="New conversation">
            <RotateCcw className="h-4 w-4" />
          </button>
        )}
      </header>

      <div ref={scroller} className="flex-1 space-y-5 overflow-y-auto px-5 py-6">
        <div className="flex gap-3">
          <Avatar name={persona.name} size="sm" />
          <div className="max-w-[80%] rounded-2xl rounded-tl-sm bg-surface-2 px-4 py-2.5 text-sm leading-relaxed text-fg-2">{persona.greeting}</div>
        </div>
        {messages.map((m, i) => (
          <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className={clsx("flex gap-3", m.role === "user" && "justify-end")}>
            {m.role === "assistant" && <Avatar name={persona.name} size="sm" />}
            <div className="max-w-[80%]">
              <div className={clsx("rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
                                   m.role === "user" ? "rounded-tr-sm bg-blood-500 text-white" : "rounded-tl-sm bg-surface-2 text-fg",
                                   m.error && "border border-blood-500/40 bg-blood-500/10 text-blood-300")}>
                {m.content || (
                  <span className="flex gap-1 py-1.5">{[0, 1, 2].map((d) => (
                    <motion.span key={d} className="h-1.5 w-1.5 rounded-full bg-fg-3" animate={{ opacity: [0.3, 1, 0.3] }}
                                 transition={{ duration: 1, repeat: Infinity, delay: d * 0.15 }} />))}</span>
                )}
              </div>
              {m.retrieved && <Memories r={m.retrieved} />}
            </div>
          </motion.div>
        ))}
      </div>

      <div className="border-t border-line p-4">
        {messages.length === 0 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {(PROMPTS[persona.name] ?? []).map((p) => (
              <button key={p} onClick={() => void send(p)} className="rounded-full border border-line px-3 py-1 text-xs text-fg-2 transition hover:border-blood-500/40 hover:text-fg">{p}</button>
            ))}
          </div>
        )}
        <form onSubmit={submit} className="flex items-center gap-2 rounded-full border border-line bg-surface-2 py-1.5 pl-5 pr-1.5 transition focus-within:border-blood-500/50 focus-within:ring-2 focus-within:ring-blood-500/15">
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder={`Message ${persona.name}…`} className="flex-1 bg-transparent text-sm text-fg outline-none placeholder:text-fg-3" />
          {busy ? (
            <button type="button" onClick={() => abort.current?.abort()} className="grid h-9 w-9 place-items-center rounded-full bg-white/10 text-fg"><Square className="h-3.5 w-3.5" /></button>
          ) : (
            <button type="submit" disabled={!input.trim()} className="grid h-9 w-9 place-items-center rounded-full bg-blood-500 text-white transition hover:bg-blood-400 disabled:opacity-40"><SendHorizontal className="h-4 w-4" /></button>
          )}
        </form>
      </div>
    </div>
  );
}

export function Chat() {
  const personas = useAsync(api.personas, []);
  const [active, setActive] = useState("Naruto");
  const persona = personas.data?.find((p) => p.name === active) ?? personas.data?.[0];
  return (
    <Section id="chat" eyebrow="04 · Retrieval-augmented LLM personas" title={<>Copy a <span className="text-gradient-blood">shinobi</span></>}
             lead={<>Each character is a card (identity, personality, speech) plus two retrieval memories: lines the character
               really said, from the transcript and from catchphrase-matched subtitles, and scenes from all 220 episodes.
               LoRA adapters are supported, but only switched on if they pass an A/B check against the base model.</>}>
      {!personas.data ? (personas.loading ? <Spinner /> : <NotReady error={personas.error} command="sharingan app" />) : (
        <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
          <div className="grid content-start gap-2 sm:grid-cols-2 lg:grid-cols-1">
            {personas.data.map((p) => (
              <button key={p.name} onClick={() => setActive(p.name)}
                      className={clsx("glass group flex items-center gap-3 rounded-2xl p-3 text-left transition",
                                      p.name === persona?.name ? "!border-blood-500/50 shadow-[0_0_30px_-10px_rgba(229,56,59,0.6)]" : "hover:!border-white/15")}>
                <Avatar name={p.name} />
                <div className="min-w-0">
                  <p className="flex items-center gap-2 font-medium">{p.fullName}
                    {p.hasAdapter && <span className="rounded-full bg-blood-500/15 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-blood-300">LoRA</span>}
                  </p>
                  <p className="line-clamp-2 text-xs text-fg-3">{p.identity}</p>
                </div>
              </button>
            ))}
          </div>
          {persona && <Conversation key={persona.name} persona={persona} />}
        </div>
      )}
    </Section>
  );
}
