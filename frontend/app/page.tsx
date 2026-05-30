import Link from "next/link";

const CARDS = [
  { href: "/flow", title: "Options Flow", desc: "Sweeps >= $50K, blocks >= $100K, ask-side only, ETFs excluded.", live: true },
  { href: "/signal", title: "Signal & Monte Carlo", desc: "Signal, MC bands, regime, trade setup.", live: true },
  { href: "/chart", title: "Price Chart", desc: "Candles, EMA/Bollinger/VWAP, trade levels.", live: true },
  { href: "/scanner", title: "Scanner", desc: "Breakout/breakdown scan with score bars.", live: true },
  { href: "/gex", title: "Options GEX / Max Pain", desc: "Gamma profile, call/put walls, gamma flip, max pain.", live: true },
  { title: "News & Sentiment", desc: "Streaming headlines and sentiment." },
  { title: "Backtest", desc: "Strategy backtest with costs." },
];

export default function Home() {
  return (
    <div className="pt-6">
      <h1 className="text-xl font-semibold">MC Trader</h1>
      <p className="mt-2 max-w-2xl text-sm text-muted">
        Next.js frontend for the Monte Carlo trader dashboard.
      </p>

      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {CARDS.map((c) => (
          <FeatureCard key={c.title} {...c} />
        ))}
      </div>
    </div>
  );
}

function FeatureCard({
  href,
  title,
  desc,
  live,
}: {
  href?: string;
  title: string;
  desc: string;
  live?: boolean;
}) {
  const inner = (
    <div
      className={
        "h-full rounded-xl border border-line bg-panel p-4 transition-colors hover:border-line2 " +
        (href ? "" : "opacity-70")
      }
    >
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">{title}</h2>
        {live ? (
          <span className="rounded-full border border-up/25 bg-up/10 px-2 py-0.5 text-[10px] text-up">
            Live
          </span>
        ) : (
          <span className="rounded-full border border-line px-2 py-0.5 text-[10px] text-dim">
            Planned
          </span>
        )}
      </div>
      <p className="mt-1.5 text-[12px] leading-relaxed text-muted">{desc}</p>
    </div>
  );

  return href ? (
    <Link href={href} className="block">
      {inner}
    </Link>
  ) : (
    inner
  );
}
