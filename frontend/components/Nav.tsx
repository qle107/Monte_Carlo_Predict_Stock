"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/chart", label: "Chart" },
  { href: "/flow", label: "Options Flow" },
  { href: "/gex", label: "GEX" },
  { href: "/scanner", label: "Scanner" },
  { href: "/signal", label: "Signal & MC" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-bg/90 backdrop-blur">
      <div className="mx-auto flex max-w-[1480px] items-center gap-5 px-4 py-2.5">
        <span className="text-[15px] font-bold tracking-tight">
          MC<span className="text-blue">Trader</span>
        </span>
        <nav className="flex items-center gap-1">
          {LINKS.map((l) => {
            const active = path === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={
                  "rounded-md px-3 py-1.5 text-[13px] transition-colors " +
                  (active
                    ? "bg-[#161c23] text-ink"
                    : "text-muted hover:text-ink")
                }
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
