/* Icons from Lucide (https://lucide.dev), ISC licence, inlined so the app
   pulls no icon package and works offline.

   Each entry is the inner geometry on a 24x24 grid, exported on its own so it
   can be dropped into a <g> inside the flow diagram as well as wrapped in a
   standalone <svg>. Strokes inherit currentColor. */

import type { ReactNode } from "react";

export type IconName =
  | "question"
  | "route"
  | "refuse"
  | "database"
  | "inject"
  | "generate"
  | "split"
  | "semantic"
  | "numbers"
  | "verdict"
  | "answer"
  | "reply";

export const ICONS: Record<IconName, ReactNode> = {
  question: (
    <>
      <path d="M2.992 16.342a2 2 0 0 1 .094 1.167l-1.065 3.29a1 1 0 0 0 1.236 1.168l3.413-.998a2 2 0 0 1 1.099.092 10 10 0 1 0-4.777-4.719" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <path d="M12 17h.01" />
    </>
  ),
  route: (
    <>
      <path d="M16 3h5v5" />
      <path d="M8 3H3v5" />
      <path d="M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3" />
      <path d="m15 9 6-6" />
    </>
  ),
  refuse: (
    <>
      <circle cx="12" cy="12" r="10" />
      <line x1="9" x2="15" y1="15" y2="9" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5V19A9 3 0 0 0 21 19V5" />
      <path d="M3 12A9 3 0 0 0 21 12" />
    </>
  ),
  inject: (
    <>
      <path d="M4 11V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.706.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-1" />
      <path d="M14 2v5a1 1 0 0 0 1 1h5" />
      <path d="M2 15h10" />
      <path d="m9 18 3-3-3-3" />
    </>
  ),
  generate: (
    <>
      <path d="M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z" />
      <path d="M20 2v4" />
      <path d="M22 4h-4" />
    </>
  ),
  split: (
    <>
      <path d="M8 5h13" />
      <path d="M13 12h8" />
      <path d="M13 19h8" />
      <path d="M3 10a2 2 0 0 0 2 2h3" />
      <path d="M3 5v12a2 2 0 0 0 2 2h3" />
    </>
  ),
  semantic: (
    <>
      <path d="M3 7V5a2 2 0 0 1 2-2h2" />
      <path d="M17 3h2a2 2 0 0 1 2 2v2" />
      <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
      <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
      <circle cx="12" cy="12" r="3" />
      <path d="m16 16-1.9-1.9" />
    </>
  ),
  numbers: (
    <>
      <rect x="14" y="14" width="4" height="6" rx="2" />
      <rect x="6" y="4" width="4" height="6" rx="2" />
      <path d="M6 20h4" />
      <path d="M14 10h4" />
      <path d="M6 14h2v6" />
      <path d="M14 4h2v6" />
    </>
  ),
  verdict: (
    <>
      <circle cx="12" cy="12" r="10" />
      <path d="m9 12 2 2 4-4" />
    </>
  ),
  answer: (
    <>
      <path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z" />
      <path d="M14 2v5a1 1 0 0 0 1 1h5" />
      <path d="m9 15 2 2 4-4" />
    </>
  ),
  reply: (
    <>
      <path d="M22 17a2 2 0 0 1-2 2H6.828a2 2 0 0 0-1.414.586l-2.202 2.202A.71.71 0 0 1 2 21.286V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2z" />
      <path d="M7 11h10" />
      <path d="M7 15h6" />
      <path d="M7 7h8" />
    </>
  ),
};

/** Drop an icon into an existing SVG at (x, y), drawn at `size` px. */
export function GlyphAt({
  name,
  x,
  y,
  size = 20,
}: {
  name: IconName;
  x: number;
  y: number;
  size?: number;
}) {
  const scale = size / 24;
  return (
    <g
      transform={`translate(${x} ${y}) scale(${scale})`}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6 / scale}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {ICONS[name]}
    </g>
  );
}
