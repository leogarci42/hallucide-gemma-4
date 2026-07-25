import type { Metadata, Viewport } from "next";
import { THEME_BOOT_SCRIPT } from "@/lib/theme";
import "./globals.css";

export const metadata: Metadata = {
  title: "Alien Hallucination",
  description:
    "Answers from a local Gemma 4, with every claim checked word for word against the passage it came from.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#000000" },
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT_SCRIPT }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
