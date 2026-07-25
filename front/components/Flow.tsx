"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { GlyphAt, type IconName } from "./icons";
import styles from "./flow.module.css";

/* The pipeline, end to end, with the question travelling through it.
   Nine steps laid out as a serpentine so the whole path fits one screen.
   Motion is declarative SVG (animateMotion + mpath), no animation library. */

const W = 216;
const H = 84;
const ROW_A = 40;
const ROW_B = 476;
const COLS = [24, 328, 632, 936, 1240];
// the lower row has four boxes; spread them over the same span as the upper one
const BCOLS = [24, 429, 835, 1240];
const DS_Y = 236;
const DS_H = 38;
const DS_W = 152;
const DS_GAP = 22;
const LANE_OFFSET = 100;
// the dataset row is centred under the retrieval step
const DS_X0 = 632 + 216 / 2 - (2 * 152 + 22) / 2;

type Node = {
  id: string;
  step: string;
  title: string;
  sub?: string;
  icon: IconName;
  x: number;
  y: number;
  w?: number;
  h?: number;
  tone?: "muted" | "output";
  owner?: "gemma" | "alien";
  /** Shown when the step is opened. */
  detail: string;
};

const NODES: Node[] = [
  { id: "ask", step: "1", title: "User prompt", sub: "the question as typed", icon: "question", x: COLS[0], y: ROW_A, detail: "The question exactly as typed. Nothing leaves the browser until you ask." },
  { id: "route", step: "2", title: "Routing", sub: "Gemma picks a dataset", icon: "route", x: COLS[1], y: ROW_A, owner: "gemma", detail: "Gemma is handed the closed list of available datasets and asked which one covers the question. It answers with a single name, or none. A code guard then checks that the answer really is in the list, so a name the model invented is treated the same as none." },
  { id: "search", step: "3", title: "Alien search", sub: "many passages, that dataset", icon: "database", x: COLS[2], y: ROW_A, owner: "alien", detail: "Semantic search inside the chosen dataset, and only that one. It returns many candidate passages rather than a single best match, so the generation step has real material to work from." },
  { id: "inject", step: "4", title: "Context injection", sub: "retrieved passages into Gemma", icon: "inject", x: COLS[3], y: ROW_A, owner: "gemma", detail: "The retrieved passages go into Gemma's prompt, together with an instruction to answer from those passages and nothing else. This is where the large context is spent." },
  { id: "draft", step: "5", title: "Generation", sub: "from those passages only", icon: "generate", x: COLS[4], y: ROW_A, owner: "gemma", detail: "Gemma writes an answer from the passages it was given. Nothing has been checked at this point: this is the draft, and it is kept so you can compare it with what survives the checks." },

  { id: "refuse", step: "2b", title: "None, or off-list", sub: "code guard, pipeline ends", icon: "refuse", x: COLS[1], y: 248, tone: "muted", detail: "Either Gemma answered none, or its answer was not in the list. The pipeline stops here and the question is refused. Nothing is generated, so there is nothing to hallucinate." },

  { id: "split", step: "6", title: "Self-decomposition", sub: "one sentence, one claim", icon: "split", x: BCOLS[3], y: ROW_B, owner: "gemma", detail: "Gemma re-reads its own answer in the same context window and cuts it into elementary claims, one sentence each. Splitting inside the same window keeps the claims in the wording the model actually used." },
  { id: "laneA", step: "7a", title: "Semantic checking", sub: "score above threshold", icon: "semantic", x: BCOLS[2], y: ROW_B - LANE_OFFSET, detail: "Each claim is compared to the source passages by semantic similarity. It passes this lane when the score clears the threshold." },
  { id: "laneB", step: "7b", title: "Literal checking", sub: "figures and negations", icon: "numbers", x: BCOLS[2], y: ROW_B + LANE_OFFSET, detail: "A deterministic pass over the same claim: figures, quantities and negations have to line up with the source. No model judgement here, only comparison." },
  { id: "verdict", step: "8", title: "Deterministic|aggregation", sub: "computes the score", icon: "verdict", x: BCOLS[1], y: ROW_B, detail: "A claim is valid only if both lanes pass. Failing either makes it a hallucination. A lane that cannot settle it leaves the claim unverifiable, which is stated rather than rounded up to fine. The verdict comes from the source, never from the model judging its own work." },
  { id: "out", step: "9", title: "Output", sub: "each claim with its source", icon: "reply", x: BCOLS[0], y: ROW_B, tone: "output", detail: "The answer comes back annotated claim by claim. Each one carries its verdict, backed, contradicted or unverifiable, and the source passage it was checked against sits beside it, so any sentence can be traced to the text it came from. The footer states how much context was injected: how many passages, how many tokens, and which dataset." },
];

/* Placeholders on purpose: the diagram should not name corpora that may not
   be indexed. Swap in the real ones once they are. */
const DATASETS = ["OpenAIRE Research", "medRxiv"];

const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));
const right = (n: Node) => ({ x: n.x + (n.w ?? W), y: n.y + (n.h ?? H) / 2 });
const left = (n: Node) => ({ x: n.x, y: n.y + (n.h ?? H) / 2 });
const bottom = (n: Node) => ({ x: n.x + (n.w ?? W) / 2, y: n.y + (n.h ?? H) });
const top = (n: Node) => ({ x: n.x + (n.w ?? W) / 2, y: n.y });

type P = { x: number; y: number };

/** S-curve segment, appended to an open path (no leading M). */
const hSeg = (a: P, b: P) => {
  const mid = (a.x + b.x) / 2;
  return ` C${mid},${a.y} ${mid},${b.y} ${b.x},${b.y}`;
};
const vSeg = (a: P, b: P) => {
  const mid = (a.y + b.y) / 2;
  return ` C${a.x},${mid} ${b.x},${mid} ${b.x},${b.y}`;
};

const hCurve = (a: P, b: P) => `M${a.x},${a.y}` + hSeg(a, b);
const vCurve = (a: P, b: P) => `M${a.x},${a.y}` + vSeg(a, b);

type Edge = { id: string; d: string; dashed?: boolean };

/* Drawn wires: what the reader sees at rest. */
const EDGES: Edge[] = [
  { id: "e1", d: hCurve(right(byId.ask), left(byId.route)) },
  { id: "e2", d: hCurve(right(byId.route), left(byId.search)) },
  { id: "e3", d: hCurve(right(byId.search), left(byId.inject)) },
  { id: "e4", d: hCurve(right(byId.inject), left(byId.draft)) },
  { id: "e5", d: vCurve(bottom(byId.draft), top(byId.split)) },
  { id: "e6a", d: hCurve(left(byId.split), right(byId.laneA)) },
  { id: "e6b", d: hCurve(left(byId.split), right(byId.laneB)) },
  { id: "e7a", d: hCurve(left(byId.laneA), right(byId.verdict)) },
  { id: "e7b", d: hCurve(left(byId.laneB), right(byId.verdict)) },
  { id: "e8", d: hCurve(left(byId.verdict), right(byId.out)) },
  { id: "eref", d: vCurve(bottom(byId.route), top(byId.refuse)), dashed: true },
];

/* One continuous path per journey, so a single dot carries the question the
   whole way instead of relay-racing between segments. Crossing a box is an
   explicit straight segment, which is what makes the dot look like it is
   being handled rather than teleporting. */
const across = (n: Node, dir: "ltr" | "rtl") =>
  dir === "ltr" ? ` L${right(n).x},${right(n).y}` : ` L${left(n).x},${left(n).y}`;

const enter = (n: Node, dir: "ltr" | "rtl") => (dir === "ltr" ? left(n) : right(n));
const exit = (n: Node, dir: "ltr" | "rtl") => (dir === "ltr" ? right(n) : left(n));

type Step = { node: Node; dir: "ltr" | "rtl" };

/** `units` counts box crossings and hops, so durations can be set from it and
    every dot moves at roughly the same speed whatever route it takes. */
function journey(
  steps: Step[],
  opts: { vAt?: number; stopAtEntry?: boolean; startAtExit?: boolean } = {},
): { d: string; units: number } {
  const first = steps[0];
  const start = opts.startAtExit ? exit(first.node, first.dir) : enter(first.node, first.dir);
  let d = `M${start.x},${start.y}`;
  let units = 0;

  const last = steps.length - 1;
  if (!opts.startAtExit) {
    d += across(first.node, first.dir);
    units += 1;
  }

  for (let i = 1; i < steps.length; i++) {
    const prev = steps[i - 1];
    const cur = steps[i];
    // the hop out of the last box on the top row drops a row instead of
    // running sideways
    if (i === opts.vAt) {
      // the previous box was crossed to its side; step down to its underside
      // first, so the drop runs along the same straight wire that is drawn
      d += ` L${bottom(prev.node).x},${exit(prev.node, prev.dir).y}`;
      d += ` L${bottom(prev.node).x},${bottom(prev.node).y}`;
      d += vSeg(bottom(prev.node), top(cur.node));
      units += 2;
    } else {
      d += hSeg(exit(prev.node, prev.dir), enter(cur.node, cur.dir));
      units += 1;
    }
    if (i === last && opts.stopAtEntry) break;
    d += across(cur.node, cur.dir);
    units += 1;
  }

  return { d, units };
}

/* The question stops at the retrieval step, the corpora feed it, and only then
   does it carry on. Legs are timed off `units` so nothing drifts out of step. */
const LEG_IN_A = journey(
  [
    { node: byId.ask, dir: "ltr" },
    { node: byId.route, dir: "ltr" },
    { node: byId.search, dir: "ltr" },
  ],
  { stopAtEntry: true },
);

const LEG_IN_B = journey(
  [
    { node: byId.search, dir: "ltr" },
    { node: byId.inject, dir: "ltr" },
    { node: byId.draft, dir: "ltr" },
    { node: byId.split, dir: "rtl" },
  ],
  { vAt: 3 },
);

const LEG_A = journey(
  [
    { node: byId.split, dir: "rtl" },
    { node: byId.laneA, dir: "rtl" },
    { node: byId.verdict, dir: "rtl" },
  ],
  { startAtExit: true, stopAtEntry: true },
);

const LEG_B = journey(
  [
    { node: byId.split, dir: "rtl" },
    { node: byId.laneB, dir: "rtl" },
    { node: byId.verdict, dir: "rtl" },
  ],
  { startAtExit: true, stopAtEntry: true },
);

const LEG_OUT = journey([
  { node: byId.verdict, dir: "rtl" },
  { node: byId.out, dir: "rtl" },
]);

/* Each corpus feeding the retrieval step. */
const FEEDS = DATASETS.map((_, i) => {
  const x = DS_X0 + i * (DS_W + DS_GAP) + DS_W / 2;
  const from = { x, y: DS_Y };
  const to = bottom(byId.search);
  return { id: `feed-${i}`, d: `M${from.x},${from.y}` + vSeg(from, to) };
});

/* One colour per transformation of the original input. Red and green are
   reserved for the verdict, so nothing before it can be mistaken for one. */
const C_QUESTION = "#ffffff";
/* Literal hex, not var(): SMIL does not resolve custom properties inside an
   animation's values list, and an unresolvable value silently disables the
   whole animation. */
const C_CONTEXT = "#56e4e6";
const C_DRAFT = "#f2c14e";
const C_CLAIM_A = "#a78bfa";
const C_CLAIM_B = "#4ea8de";
const C_BACKED = "#7fbf8f";
const C_WITHHELD = "#e06a6a";
const C_UNCHECKED = "#868686"; // neither check could settle it

/* SMIL has no randomness, so each claim carries its own sequence of outcomes
   and the sequences are different lengths: the five never settle into the same
   arrangement twice running. */
const OUTCOMES = [
  [C_BACKED, C_BACKED, C_WITHHELD],
  [C_BACKED, C_UNCHECKED],
  [C_WITHHELD, C_BACKED, C_BACKED, C_BACKED],
  [C_BACKED, C_UNCHECKED, C_BACKED, C_WITHHELD, C_BACKED],
  [C_UNCHECKED, C_BACKED, C_BACKED],
];


const PER_UNIT = 0.9; // seconds
const secs = (u: number) => u * PER_UNIT;

/* The question reaches retrieval at 4 units. The corpora only set off once it
   is there, so the feeding reads as a consequence rather than a coincidence. */
const T_FEED = secs(4.6);
const FEED_DUR = secs(1.5);
const T_RESUME = secs(6.4);
const T_LANES = T_RESUME + secs(LEG_IN_B.units);
const T_OUT = T_LANES + secs(LEG_A.units);
const LAP = T_OUT + secs(LEG_OUT.units) + 2.6;

/** How much of a lap a box stays lit: long enough to read, then dark again. */
const PULSE_SECONDS = PER_UNIT;

/** When a dot is inside a box, and the colour that box passes on. */
const PULSE = PULSE_SECONDS / LAP;

const HIGHLIGHT: Record<string, { at: number; tone: string; tone2?: string }> = {
  ask: { at: secs(0), tone: C_QUESTION },
  route: { at: secs(2), tone: C_QUESTION },
  search: { at: T_RESUME, tone: C_QUESTION },
  inject: { at: T_RESUME + secs(2), tone: C_QUESTION },
  draft: { at: T_RESUME + secs(4), tone: C_DRAFT },
  // split hands on two claims, so it lights in both lanes' colours
  split: { at: T_RESUME + secs(7), tone: C_CLAIM_A, tone2: C_CLAIM_B },
  laneA: { at: T_LANES + secs(1), tone: C_CLAIM_A },
  laneB: { at: T_LANES + secs(1), tone: C_CLAIM_B },
  verdict: { at: T_OUT, tone: C_BACKED, tone2: C_WITHHELD },
  out: { at: T_OUT + secs(2), tone: C_QUESTION },
};

/* Every rider shares one lap so the stages stay in step. `dur` is how long
   this leg takes; the dot is hidden for the rest of the lap. */
function Rider({
  path,
  tone,
  begin,
  dur,
  lap,
  becomes = [],
  r = 5,
  growsTo,
  growAt = 0.14,
  cycleTones,
}: {
  path: string;
  tone: string;
  begin: number;
  dur: number;
  lap: number;
  /** The input changes hands along some legs. `at` is the fraction of this leg
      where it does; the switch is discrete, so the colour changes on the spot
      rather than fading into the next one. */
  becomes?: { tone: string; at: number }[];
  r?: number;
  /** The dot carries more once a step has added to it, so it gets bigger.
      `growAt` is the fraction of the leg where that happens. */
  growsTo?: number;
  growAt?: number;
  /** A colour per lap, cycled. SMIL has no randomness; a short irregular
      cycle reads as one without needing script. */
  cycleTones?: string[];
}) {
  const f = dur / lap;
  const values = [tone, ...becomes.map((b) => b.tone)];
  const times = [0, ...becomes.map((b) => f * b.at)];
  return (
    <circle className={styles.dot} r={r} fill={tone} opacity="0">
      {cycleTones && cycleTones.length > 1 && (
        <animate
          attributeName="fill"
          calcMode="discrete"
          values={[...cycleTones, cycleTones[cycleTones.length - 1]].join(";")}
          keyTimes={[...cycleTones.map((_, i) => i / cycleTones.length), 1]
            .map((t) => t.toFixed(4))
            .join(";")}
          dur={`${lap * cycleTones.length}s`}
          begin={`${begin}s`}
          repeatCount="indefinite"
        />
      )}
      {growsTo !== undefined && (
        <animate
          attributeName="r"
          values={`${r};${r};${growsTo};${growsTo}`}
          keyTimes={`0;${(f * growAt).toFixed(4)};${(f * (growAt + 0.16)).toFixed(4)};1`}
          dur={`${lap}s`}
          begin={`${begin}s`}
          repeatCount="indefinite"
        />
      )}
      {becomes.length > 0 && (
        <animate
          attributeName="fill"
          calcMode="discrete"
          values={[...values, values[values.length - 1]].join(";")}
          keyTimes={[...times, 1].map((t) => t.toFixed(4)).join(";")}
          dur={`${lap}s`}
          begin={`${begin}s`}
          repeatCount="indefinite"
        />
      )}
      <animateMotion
        dur={`${lap}s`}
        begin={`${begin}s`}
        repeatCount="indefinite"
        calcMode="linear"
        keyPoints={`0;1;1`}
        keyTimes={`0;${f.toFixed(4)};1`}
      >
        <mpath href={`#${path}`} />
      </animateMotion>
      <animate
        attributeName="opacity"
        values="0;1;1;0;0"
        keyTimes={`0;0.004;${(f * 0.96).toFixed(4)};${(f * 0.995).toFixed(4)};1`}
        dur={`${lap}s`}
        begin={`${begin}s`}
        repeatCount="indefinite"
      />
    </circle>
  );
}

const OWNER_CLASS = { gemma: styles.ownerGemma, alien: styles.ownerAlien } as const;

function Box({ node, onOpen }: { node: Node; onOpen: (n: Node) => void }) {
  const w = node.w ?? W;
  const h = node.h ?? H;
  const hit = HIGHLIGHT[node.id];
  const cls = [
    node.tone === "muted" ? styles.boxMuted : node.tone === "output" ? styles.boxOutput : styles.box,
    node.owner ? OWNER_CLASS[node.owner] : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <g
      className={`${cls} ${styles.hit}`}
      role="button"
      tabIndex={0}
      aria-label={`${node.title.replace("|", " ")}, step ${node.step}`}
      onClick={() => onOpen(node)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen(node);
        }
      }}
    >
      {/* the plate behind, offset up and to the right so it reads as a card
          sitting on another card, the way alien.club draws them */}
      <rect
        x={node.x + 7}
        y={node.y - 7}
        width={w}
        height={h}
        rx="3"
        className={styles.plateBack}
        stroke={node.owner === "gemma" ? `url(#sweep-${node.id})` : "var(--line-soft)"}
      />
      <rect
        x={node.x}
        y={node.y}
        width={w}
        height={h}
        rx="3"
        className={styles.plate}
        stroke={
          node.owner === "gemma"
            ? `url(#sweep-${node.id})`
            : node.owner === "alien"
              ? "var(--alien)"
              : "var(--line)"
        }
      />
      {hit?.tone2 && (
        <rect
          x={node.x + 7}
          y={node.y - 7}
          width={w}
          height={h}
          rx="3"
          className={styles.lit}
          stroke={hit.tone2}
          opacity="0"
        >
          <animate
            attributeName="opacity"
            values="0;1;1;0;0"
            keyTimes={`0;${(PULSE * 0.2).toFixed(4)};${(PULSE * 0.65).toFixed(4)};${PULSE.toFixed(4)};1`}
            dur={`${LAP}s`}
            begin={`${hit.at}s`}
            repeatCount="indefinite"
          />
        </rect>
      )}
      {hit && (
        <rect
          x={node.x}
          y={node.y}
          width={w}
          height={h}
          rx="3"
          className={styles.lit}
          stroke={hit.tone}
          opacity="0"
        >
          {/* one pulse per lap, timed to the dot passing through */}
          <animate
            attributeName="opacity"
            values="0;1;1;0;0"
            keyTimes={`0;${(PULSE * 0.2).toFixed(4)};${(PULSE * 0.65).toFixed(4)};${PULSE.toFixed(4)};1`}
            dur={`${LAP}s`}
            begin={`${hit.at}s`}
            repeatCount="indefinite"
          />
        </rect>
      )}
      <text className={styles.step} x={node.x + 14} y={node.y + 21}>
        {node.step}
      </text>
      <GlyphAt name={node.icon} x={node.x + w - 40} y={node.y + 12} size={22} />
      {/* a pipe in the title is a line break: SVG text does not wrap */}
      <text
        className={node.title.includes("|") ? styles.titleTwo : styles.title}
        x={node.x + w / 2}
        y={node.y + h - (node.sub ? 28 : 18) - (node.title.includes("|") ? 15 : 0)}
        textAnchor="middle"
      >
        {node.title.split("|").map((line, i) => (
          <tspan key={i} x={node.x + w / 2} dy={i === 0 ? 0 : 16}>
            {line}
          </tspan>
        ))}
      </text>
      {node.sub && (
        <text className={styles.sub} x={node.x + w / 2} y={node.y + h - 12} textAnchor="middle">
          {node.sub}
        </text>
      )}
    </g>
  );
}

export default function Flow() {
  const v = byId.verdict;
  const [open, setOpen] = useState<Node | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation(); // the page also listens for Escape
        setOpen(null);
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open]);

  return (
    <div className={styles.wrap}>
      <svg
        className={styles.svg}
        viewBox="0 0 1480 680"
        role="img"
        aria-label="A question is routed to a dataset, passages are retrieved and injected into Gemma's prompt, the draft is split into claims, each claim is checked semantically and literally, then the answer comes back annotated claim by claim."
      >
        <defs>
          {/* Google's four brand colours, swept around the plate. The rotation
              is what makes a Gemma step read as the live part of the diagram. */}
          {/* Google's four colours. One gradient per Gemma box, in user space,
              so turning it sweeps the bands round the border instead of being
              squashed by the box aspect ratio. */}
          {NODES.filter((n) => n.owner === "gemma").map((n) => {
            const w = n.w ?? W;
            const h = n.h ?? H;
            const cx = n.x + w / 2;
            const cy = n.y + h / 2;
            return (
              <linearGradient
                key={n.id}
                id={`sweep-${n.id}`}
                gradientUnits="userSpaceOnUse"
                x1={n.x}
                y1={cy}
                x2={n.x + w}
                y2={cy}
              >
                <stop offset="0" stopColor="#4285F4" />
                <stop offset="0.25" stopColor="#EA4335" />
                <stop offset="0.5" stopColor="#FBBC04" />
                <stop offset="0.75" stopColor="#34A853" />
                <stop offset="1" stopColor="#4285F4" />
                <animateTransform
                  attributeName="gradientTransform"
                  type="rotate"
                  from={`0 ${cx} ${cy}`}
                  to={`360 ${cx} ${cy}`}
                  dur="6s"
                  repeatCount="indefinite"
                />
              </linearGradient>
            );
          })}
          {EDGES.map((e) => (
            <path key={e.id} id={e.id} d={e.d} fill="none" />
          ))}
          <path id="leg-in-a" d={LEG_IN_A.d} fill="none" />
          <path id="leg-in-b" d={LEG_IN_B.d} fill="none" />
          {FEEDS.map((f) => (
            <path key={f.id} id={f.id} d={f.d} fill="none" />
          ))}
          <path id="leg-a" d={LEG_A.d} fill="none" />
          <path id="leg-b" d={LEG_B.d} fill="none" />
          <path id="leg-out" d={LEG_OUT.d} fill="none" />
        </defs>

        {EDGES.map((e) => (
          <use key={e.id} href={`#${e.id}`} className={e.dashed ? styles.wireDashed : styles.wire} />
        ))}

        {/* the corpora the router picks between, each wired to the retrieval
            step; the chosen one is lit */}
        <g className={styles.datasets}>
          {DATASETS.map((name, i) => {
            const x = DS_X0 + i * (DS_W + DS_GAP);
            const from = bottom(byId.search);
            return (
              <g key={name} className={styles.dataset}>
                <path
                  d={`M${from.x},${from.y}` + vSeg(from, { x: x + DS_W / 2, y: DS_Y })}
                  className={styles.wireDashed}
                />
                <rect x={x} y={DS_Y} width={DS_W} height={DS_H} rx="3" />
                <GlyphAt name="database" x={x + 12} y={DS_Y + 11} size={16} />
                <text x={x + 36} y={DS_Y + 24}>
                  {name}
                </text>
              </g>
            );
          })}
        </g>

        {/* the question, as far as the retrieval step */}
        <Rider
          path="leg-in-a"
          tone={C_QUESTION}
          begin={0}
          dur={secs(LEG_IN_A.units)}
          lap={LAP}
        />

        {/* the corpora feeding it, in Alien's colour */}
        {FEEDS.map((f) => (
          <Rider key={f.id} path={f.id} tone="var(--alien)" begin={T_FEED} dur={FEED_DUR} lap={LAP} />
        ))}

        {/* fed, it carries on, and becomes a draft once Gemma has written it */}
        <Rider
          path="leg-in-b"
          r={5}
          growsTo={11}
          growAt={0.1}
          tone={C_QUESTION}
          /* leg-in-b is 8 units: cross retrieve 0-1, inject 2-3, generate 4-5,
             drop 5-7, split 7-8. It leaves retrieval carrying the passages, so
             it takes Alien's colour there, and becomes a draft on leaving
             Generate. It does not become a claim here: the two lane dots start
             at the exact point and instant this one ends. */
          becomes={[
            { tone: C_CONTEXT, at: 1 / 8 },
            { tone: C_DRAFT, at: 5 / 8 },
          ]}
          begin={T_RESUME}
          dur={secs(LEG_IN_B.units)}
          lap={LAP}
        />

        {/* one claim, two checks, each with its own colour */}
        <Rider
          path="leg-a"
          tone={C_CLAIM_A}
          r={5}
          growsTo={8}
          growAt={0.34}
          begin={T_LANES}
          dur={secs(LEG_A.units)}
          lap={LAP}
        />
        <Rider
          path="leg-b"
          tone={C_CLAIM_B}
          r={5}
          growsTo={8}
          growAt={0.34}
          begin={T_LANES}
          dur={secs(LEG_B.units)}
          lap={LAP}
        />

        {/* five claims leave the aggregation, same size, differing only in
            outcome */}
        {OUTCOMES.map((cycle, i) => (
          <Rider
            key={i}
            path="leg-out"
            tone={cycle[0]}
            cycleTones={cycle}
            r={8}
            begin={T_OUT + i * 0.32}
            dur={secs(LEG_OUT.units)}
            lap={LAP}
          />
        ))}

        {NODES.map((n) => (
          <Box key={n.id} node={n} onOpen={setOpen} />
        ))}

        {/* the three outcomes a claim can be given */}
        <g>
          {[C_BACKED, C_WITHHELD, C_UNCHECKED].map((tone, i) => (
            <circle key={tone} cx={v.x + W / 2 - 14 + i * 14} cy={v.y + H + 16} r="3.5" fill={tone} />
          ))}
        </g>



      </svg>

      {/* rendered on the body so it can never take part in the diagram's
          layout, whatever the wrap happens to be doing */}
      {open &&
        createPortal(
        <div className={styles.scrim} onClick={() => setOpen(null)}>
          <div
            className={styles.sheet}
            role="dialog"
            aria-modal="true"
            aria-label={open.title.replace("|", " ")}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className={styles.sheetX}
              onClick={() => setOpen(null)}
              aria-label="Close"
              autoFocus
            >
              &#215;
            </button>

            <p className={styles.sheetStep}>Step {open.step}</p>
            <h2 className={styles.sheetTitle}>{open.title.replace("|", " ")}</h2>
            {open.sub && <p className={styles.sheetSub}>{open.sub}</p>}
            <p className={styles.sheetBody}>{open.detail}</p>
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}
