import clsx from "clsx";

/** Animated three-tomoe Sharingan, drawn in SVG. */
export function SharinganEye({ className, spin = "slow", glow = true }: {
  className?: string; spin?: "slow" | "medium" | "none"; glow?: boolean;
}) {
  const tomoe = (
    <g>
      <circle cx="50" cy="26" r="6.2" />
      <path d="M55.6 23.6c4.8 3.4 6.4 10.6 1.4 16.2 1.6-5.6-.6-9.6-6-11.4z" />
    </g>
  );
  return (
    <div className={clsx("relative", className)}>
      {glow && <div className="absolute inset-[-18%] rounded-full bg-blood-600/40 blur-3xl animate-pulse-glow" />}
      <svg viewBox="0 0 100 100" className="relative h-full w-full drop-shadow-[0_0_25px_rgba(229,56,59,0.45)]">
        <defs>
          <radialGradient id="iris" cx="50%" cy="45%" r="60%">
            <stop offset="0%" stopColor="#ff4d5a" />
            <stop offset="55%" stopColor="#c1121f" />
            <stop offset="100%" stopColor="#5c0810" />
          </radialGradient>
          <radialGradient id="shine" cx="38%" cy="30%" r="35%">
            <stop offset="0%" stopColor="#fff" stopOpacity=".35" />
            <stop offset="100%" stopColor="#fff" stopOpacity="0" />
          </radialGradient>
        </defs>
        <circle cx="50" cy="50" r="47" fill="url(#iris)" stroke="#0a0a0c" strokeWidth="2.5" />
        <circle cx="50" cy="50" r="24" fill="none" stroke="#0a0a0c" strokeOpacity=".75" strokeWidth="1.6" />
        {/* Native SVG animation: rotates in every browser, independent of CSS transform-box
            support and of the reduced-motion override (the eye is the brand mark, not UI motion). */}
        <g fill="#0a0a0c">
          {spin !== "none" && (
            <animateTransform attributeName="transform" type="rotate" from="0 50 50" to="360 50 50"
                              dur={spin === "slow" ? "18s" : "6s"} repeatCount="indefinite" />
          )}
          {tomoe}
          <g transform="rotate(120 50 50)">{tomoe}</g>
          <g transform="rotate(240 50 50)">{tomoe}</g>
        </g>
        <circle cx="50" cy="50" r="9.5" fill="#0a0a0c" />
        <circle cx="50" cy="50" r="47" fill="url(#shine)" />
      </svg>
    </div>
  );
}
