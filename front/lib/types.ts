/* Contract between the front and the verification backend.
   Nothing here is invented at display time: every field the UI shows
   comes from the engine. If the engine cannot supply it, the UI says so
   rather than filling a gap. */

export type Verdict =
  | "grounded" // every claim cleared both checks
  | "partial" // some cleared, some did not
  | "unsupported" // none cleared
  | "refused"; // routing found no dataset covering the question

/* A claim is valid only when both lanes pass. Failing the semantic or the
   deterministic check makes it a hallucination; being unable to run either
   makes it unverifiable, which is stated rather than hidden. */
export type ClaimStatus = "grounded" | "hallucinated" | "unverifiable";

export interface Source {
  id: string;
  /** Human label, e.g. "Code civil, article 1103". */
  title: string;
  /** Deep link to the source. Absent when the engine has no public URL. */
  url?: string;
  /** The passage the claim was matched against, verbatim. */
  passage?: string;
}

export interface Claim {
  id: string;
  text: string;
  status: ClaimStatus;
  /** The passage the claim was matched against. Present when there is one. */
  source?: Source;
  /** Per-lane outcome, when the engine reports it. */
  semanticPass?: boolean;
  literalPass?: boolean;
}

export interface Answer {
  /** What the model wrote, untouched. */
  draft: string;
  /** The same answer after unsupported claims were withheld. */
  verified: string;
  claims: Claim[];
  verdict: Verdict;
  sources: Source[];
  /** Engine-reported, optional. The UI never computes or estimates these. */
  model?: string;
  /** How much was injected. Engine-reported; never estimated here. */
  contextPassages?: number;
  /** The engine counts characters, not tokens; both are shown as reported. */
  contextChars?: number;
  contextTokens?: number;
  latencyMs?: number;
  /** The dataset routing picked, when the engine names it. */
  dataset?: string;
}

export type TurnState =
  | { phase: "pending" }
  | { phase: "done"; answer: Answer }
  | { phase: "error"; error: ApiError }
  | { phase: "cancelled" };

export interface Turn {
  id: string;
  question: string;
  askedAt: number;
  state: TurnState;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  turns: Turn[];
}

export type ApiErrorKind =
  | "offline" // browser reports no network
  | "unreachable" // engine did not answer at all
  | "timeout"
  | "server" // engine answered with 5xx
  | "bad_request" // engine answered with 4xx
  | "malformed" // engine answered with something we cannot read
  | "cancelled";

export interface ApiError {
  kind: ApiErrorKind;
  /** Shown to the user. Written in plain language, no stack traces. */
  message: string;
  /** Kept for the details panel; never shown by default. */
  detail?: string;
  status?: number;
  /** Whether asking again is worth it. */
  retryable: boolean;
  attempts: number;
}

export interface HealthReport {
  reachable: boolean;
  model?: string;
  checkedAt: number;
  error?: ApiError;
}
