import Link from "next/link";

export default function Home() {
  return (
    <div className="pt-6">
      <h1 className="text-xl font-semibold">MC Trader — Next.js frontend</h1>
      <p className="mt-2 max-w-2xl text-sm text-muted">
        Migration foundation for the Monte Carlo trader dashboard. The Options
        Flow feed is fully implemented; remaining panels (signal, chart, scanner,
        GEX) are migrated incrementally — see <code className="rounded bg-[#161c23] px-1 text-blue">README.md</code>.
      </p>

      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Card
          href="/flow"
          title="Options Flow"
          desc="Sweeps ≥ $50K, blocks ≥ $100K, ask-side conviction, ETFs excluded."
          live
        />
        <Card
          href="/signal"
          title="Signal & Monte Carlo"
          desc="Live signal, MC probability bands, regime, trade setup."
          live
        />
        <Card href="/chart" title="Price Chart" desc="Candles, EMA/Bollinger/VWAP, crosshair, trade levels." live />
        <Card href="/scanner" title="Scanner" desc="Breakout / breakdown multi-ticker scan with score bars." live />
        <Card href="/gex" title="Options GEX / Max Pain" desc="Gamma profile, call/put walls, γ-flip, max pain." live />
        <Card title="News & Sentiment" desc="Streaming headlines + sentiment." />
        <Card title="Backtest" desc="Strategy backtest with costs." />
      </div>
    </div>
  );
}

function Card({
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
    <div className="h-full rounded-xl border border-line bg-panel p-4 transition-colors hover:border-[#2c3a47]">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">{title}</h2>
        {live ? (
          <span className="rounded-full border border-[#244] bg-[#10211c] px-2 py-0.5 text-[10px] text-up">
            live
          </span>
        ) : (
          <span className="rounded-full border border-line px-2 py-0.5 text-[10px] text-dim">
            planned
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
    <div className="opacity-70">{inner}</div>
  );
}
