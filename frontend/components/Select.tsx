"use client";

import { useEffect, useRef, useState } from "react";
import { label } from "@/lib/display";

export interface SelectOption {
  value: string;
  label: string;
}

export default function Select({
  value,
  onChange,
  options,
  disabled,
  title,
  className = "",
  align = "left",
  width,
}: {
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
  disabled?: boolean;
  title?: string;
  className?: string;
  align?: "left" | "right";
  width?: number;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const current = options.find((o) => o.value === value);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
      else if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(options.length - 1, a + 1)); }
      else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(0, a - 1)); }
      else if (e.key === "Enter") { e.preventDefault(); const o = options[active]; if (o) { onChange(o.value); setOpen(false); } }
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, options, active, onChange]);

  useEffect(() => {
    if (open) {
      const i = options.findIndex((o) => o.value === value);
      setActive(i < 0 ? 0 : i);
    }
  }, [open, options, value]);

  return (
    <div ref={rootRef} className="relative" style={width ? { width } : undefined}>
      <button
        type="button"
        title={title ? label(title) : undefined}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        className={
          "field flex w-full items-center justify-between gap-2 disabled:opacity-50 " +
          (open ? "border-line2 " : "") +
          className
        }
      >
        <span className="truncate">{current?.label ?? value}</span>
        <svg
          viewBox="0 0 24 24"
          className={"h-3.5 w-3.5 shrink-0 text-muted transition-transform " + (open ? "rotate-180" : "")}
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {open && (
        <div
          role="listbox"
          className={
            "absolute z-50 mt-1 max-h-72 min-w-full overflow-auto rounded-md border border-line bg-panel p-1 shadow-panel " +
            (align === "right" ? "right-0" : "left-0")
          }
        >
          {options.map((o, i) => {
            const selected = o.value === value;
            const hot = i === active;
            return (
              <button
                key={o.value}
                type="button"
                role="option"
                aria-selected={selected}
                onMouseEnter={() => setActive(i)}
                onClick={() => { onChange(o.value); setOpen(false); }}
                className={
                  "flex w-full items-center justify-between gap-3 whitespace-nowrap rounded px-2.5 py-1.5 text-left text-[12px] " +
                  (hot ? "bg-[#161c23] text-ink " : "text-muted ") +
                  (selected ? "font-semibold text-ink" : "")
                }
              >
                <span>{o.label}</span>
                {selected && <span className="h-1.5 w-1.5 rounded-full bg-blue" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
