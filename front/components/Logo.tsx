/* The wordmark. Strokes use currentColor, so the logo is black on the light
   theme and white on the dark one without a second asset. */

export function LogoMark({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="20 4 60 66"
      fill="none"
      aria-hidden
      focusable="false"
    >
      <path
        d="M30 65 C30 30 38 10 50 10 C62 10 70 30 70 65"
        stroke="currentColor"
        strokeWidth="5"
        strokeLinecap="round"
        fill="none"
        vectorEffect="non-scaling-stroke"
      />
      <line
        x1="50"
        y1="27"
        x2="50"
        y2="47"
        stroke="currentColor"
        strokeWidth="5"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

/* viewBox trimmed to the ink, so the mark renders as large as the row allows
   instead of being padded down by the original 520-wide artboard. */
export default function Logo({ height = 34 }: { height?: number }) {
  return (
    <svg
      viewBox="22 4 358 68"
      height={height}
      fill="none"
      role="img"
      aria-label="alien hallucination"
      style={{ display: "block" }}
    >
      <path
        d="M30 65 C30 30 38 10 50 10 C62 10 70 30 70 65"
        stroke="currentColor"
        strokeWidth="4.5"
        strokeLinecap="round"
        fill="none"
      />
      <line
        x1="50"
        y1="27"
        x2="50"
        y2="47"
        stroke="currentColor"
        strokeWidth="4.5"
        strokeLinecap="round"
      />
      <text
        x="92"
        y="50"
        fill="currentColor"
        fontFamily="Dopis, system-ui, sans-serif"
        fontSize="30"
        fontWeight="500"
        letterSpacing="-0.5"
      >
        alien hallucination
      </text>
    </svg>
  );
}
