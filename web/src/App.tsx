import { MotionConfig } from "motion/react";
import { lazy, Suspense } from "react";
import { Navbar } from "./components/Navbar";
import { SharinganEye } from "./components/SharinganEye";
import { api } from "./lib/api";
import { useAsync } from "./lib/hooks";
import { Chat } from "./sections/Chat";
import { Hero } from "./sections/Hero";
import { Jutsu } from "./sections/Jutsu";
import { Themes } from "./sections/Themes";
import { Spinner } from "./components/ui";

// The force-graph engine is the heaviest dependency: load it with its section.
const Network = lazy(() => import("./sections/Network").then((m) => ({ default: m.Network })));

export default function App() {
  const overview = useAsync(api.overview, []);
  return (
    <MotionConfig reducedMotion="user">
      <Navbar />
      <main>
        <Hero overview={overview.data} />
        <Themes />
        <Suspense fallback={<Spinner label="Loading network" />}><Network /></Suspense>
        <Jutsu />
        <Chat />
      </main>
      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-4 py-10 text-sm text-fg-3 sm:flex-row sm:px-6">
          <div className="flex items-center gap-3">
            <SharinganEye className="h-6 w-6" spin="slow" glow={false} />
            <span>Sharingan v{overview.data?.version ?? "1.0.0"}: an NLP study of <em>Naruto</em> for learning purposes.</span>
          </div>
          <span>Data: subtitles · Naruto Fandom wiki (CC BY-SA) · Built with PyTorch, spaCy, Hugging Face, React</span>
        </div>
      </footer>
    </MotionConfig>
  );
}
