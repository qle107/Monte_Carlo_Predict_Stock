import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0b0e11",
        panel: "#11161c",
        rowalt: "#0c1116",
        line: "#1b232c",
        line2: "#2c3a47",
        ink: "#e6edf3",
        muted: "#7d8b99",
        dim: "#5a6673",
        up: "#3fb950",
        down: "#f0556d",
        gold: "#f2c14e",
        blue: "#58a6ff",
        sweep: "#e3b341",
        magenta: "#bc6bd9",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.03) inset",
      },
    },
  },
  plugins: [],
};

export default config;
