import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Acme Orchestrator",
  description: "Jarvis-style multi-agent voice orchestration demo.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
