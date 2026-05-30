"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/chart", label: "Chart" },
  { href: "/flow", label: "Flow" },
  { href: "/gex", label: "GEX" },
  { href: "/scanner", label: "Scanner" },
  { href: "/signal", label: "Signal" },
];

function marketState(): { label: string; open: boolean } {
  const now = new Date();
  const et = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
  const day = et.getDay();
  const mins = et.getHours() * 60 + et.getMinutes();
  const weekday = day >= 1 && day <= 5;
  if (weekday && mins >= 570 && mins < 960) return { label: "Market Open", open: true };
  if (weekday && mins >= 240 && mins < 570) return { label: "Pre-Market", open: false };
  if (weekday && mins >= 960 && mins < 1200) return { label: "After Hours", open: false };
  return { label: "Closed", open: false };
}

export default function Nav() {
  const path = usePathname();
  const [mkt, setMkt] = useState<ReturnType<typeof marketState> | null>(null);
  const [clock, setClock] = useState("");

  useEffect(() => {
    const tick = () => {
      setMkt(marketState());
      setClock(
        new Date().toLocaleTimeString("en-US", {
          timeZone: "America/New_York",
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
        })
      );
    };
    tick();
    const id = setInterval(tick, 15_000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="sticky top-0 z-20 border-b border-line bg-bg/90 backdrop-blur">
      <div className="mx-auto flex max-w-[1480px] items-center gap-4 px-4 py-2.5">
        <Link href="/" className="text-[15px] font-bold tracking-tight">
          MC<span className="text-blue">Trader</span>
        </Link>

        <nav className="flex items-center gap-1">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={
                  "rounded-md px-3 py-1.5 text-[13px] transition-colors " +
                  (active ? "bg-[#161c23] text-ink" : "text-muted hover:text-ink")
                }
              >
                {l.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex-1" />

        {mkt && (
          <div className="hidden items-center gap-2 text-[11.5px] text-muted sm:flex">
            <span
              className={
                "inline-block h-[7px] w-[7px] rounded-full " +
                (mkt.open ? "bg-up shadow-[0_0_7px_#3fb950]" : "bg-dim")
              }
            />
            <span>{mkt.label}</span>
            <span className="text-dim">|</span>
            <span className="tnum font-medium text-ink">{clock}</span>
            <span className="text-[9px] tracking-wider text-dim">ET</span>
          </div>
        )}
      </div>
    </header>
  );
}
