import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "MC Trader",
  description: "Monte Carlo trader dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg font-sans text-ink antialiased">
        <Nav />
        <main className="mx-auto max-w-[1480px] px-4 pb-16 pt-3">{children}</main>
      </body>
    </html>
  );
}
