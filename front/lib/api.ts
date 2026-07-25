import type { Answer, ApiError, ApiErrorKind, Claim, HealthReport, Source } from "./types";

const BASE = process.env.NEXT_PUBLIC_ENGINE_URL ?? "/api";

const ASK_TIMEOUT_MS = 60_000; // a local SLM on a laptop is not fast
const HEALTH_TIMEOUT_MS = 4_000;
const MAX_ATTEMPTS = 3;
const BASE_BACKOFF_MS = 400;
const MAX_BACKOFF_MS = 4_000;

const MESSAGES: Record<ApiErrorKind, string> = {
  offline: "No network connection.",
  unreachable: "The engine is not answering.",
  timeout: "The engine took too long to answer.",
  server: "The engine hit an error while answering.",
  bad_request: "The engine rejected this question.",
  malformed: "The engine answered in a format this app cannot read.",
  cancelled: "Cancelled.",
};

const RETRYABLE: ReadonlySet<ApiErrorKind> = new Set<ApiErrorKind>([
  "offline",
  "unreachable",
  "timeout",
  "server",
]);

function err(kind: ApiErrorKind, attempts: number, extra?: Partial<ApiError>): ApiError {
  return {
    kind,
    message: MESSAGES[kind],
    retryable: RETRYABLE.has(kind),
    attempts,
    ...extra,
  };
}

/** Exponential backoff with full jitter, so parallel retries do not sync up. */
function backoff(attempt: number): number {
  const ceiling = Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS * 2 ** attempt);
  return Math.random() * ceiling;
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        reject(new DOMException("aborted", "AbortError"));
      },
      { once: true },
    );
  });

/* ── response validation ────────────────────────────────────────────
   A field the engine did not send is left undefined. It is never
   defaulted to a number, because a defaulted number is a made-up one. */

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

function readSource(v: unknown): Source | undefined {
  if (!isRecord(v) || typeof v.id !== "string" || typeof v.title !== "string") return undefined;
  return {
    id: v.id,
    title: v.title,
    url: typeof v.url === "string" ? v.url : undefined,
    passage: typeof v.passage === "string" ? v.passage : undefined,
  };
}

function readClaim(v: unknown, i: number): Claim | undefined {
  if (!isRecord(v) || typeof v.text !== "string") return undefined;

  const semanticPass = typeof v.semanticPass === "boolean" ? v.semanticPass : undefined;
  const literalPass = typeof v.literalPass === "boolean" ? v.literalPass : undefined;

  /* Valid only when both lanes pass. If the engine states a status, it wins;
     otherwise it is derived from the lanes, and a lane the engine did not
     report leaves the claim unverifiable rather than assumed good. */
  let status: Claim["status"];
  if (v.status === "grounded" || v.status === "hallucinated" || v.status === "unverifiable") {
    status = v.status;
  } else if (semanticPass === undefined || literalPass === undefined) {
    status = "unverifiable";
  } else {
    status = semanticPass && literalPass ? "grounded" : "hallucinated";
  }

  return {
    id: typeof v.id === "string" ? v.id : `claim-${i}`,
    text: v.text,
    status,
    source: readSource(v.source),
    semanticPass,
    literalPass,
  };
}

function readAnswer(v: unknown): Answer | undefined {
  if (!isRecord(v)) return undefined;
  if (typeof v.draft !== "string") return undefined;

  const claims = Array.isArray(v.claims)
    ? v.claims.map(readClaim).filter((c): c is Claim => Boolean(c))
    : [];

  const verdict =
    v.verdict === "grounded" ||
    v.verdict === "partial" ||
    v.verdict === "unsupported" ||
    v.verdict === "refused"
      ? v.verdict
      : claims.length === 0
        ? "refused"
        : claims.every((c) => c.status === "grounded")
          ? "grounded"
          : claims.some((c) => c.status === "grounded")
            ? "partial"
            : "unsupported";

  return {
    draft: v.draft,
    verified: typeof v.verified === "string" ? v.verified : "",
    claims,
    verdict,
    sources: Array.isArray(v.sources)
      ? v.sources.map(readSource).filter((s): s is Source => Boolean(s))
      : [],
    model: typeof v.model === "string" ? v.model : undefined,
    dataset: typeof v.dataset === "string" ? v.dataset : undefined,
    contextPassages: typeof v.contextPassages === "number" ? v.contextPassages : undefined,
    contextTokens: typeof v.contextTokens === "number" ? v.contextTokens : undefined,
    latencyMs: typeof v.latencyMs === "number" ? v.latencyMs : undefined,
  };
}

/* ── request ──────────────────────────────────────────────────────── */

export type Result<T> = { ok: true; value: T } | { ok: false; error: ApiError };

interface RequestOptions {
  timeoutMs: number;
  maxAttempts: number;
  signal?: AbortSignal;
  /** Called before each wait, so the UI can say "retrying in a moment". */
  onRetry?: (attempt: number, error: ApiError) => void;
}

async function request<T>(
  path: string,
  init: RequestInit,
  parse: (body: unknown) => T | undefined,
  opts: RequestOptions,
): Promise<Result<T>> {
  let last: ApiError = err("unreachable", 0);

  for (let attempt = 0; attempt < opts.maxAttempts; attempt++) {
    if (opts.signal?.aborted) return { ok: false, error: err("cancelled", attempt) };

    if (typeof navigator !== "undefined" && navigator.onLine === false) {
      last = err("offline", attempt + 1);
      if (attempt < opts.maxAttempts - 1) {
        opts.onRetry?.(attempt + 1, last);
        try {
          await sleep(backoff(attempt), opts.signal);
          continue;
        } catch {
          return { ok: false, error: err("cancelled", attempt + 1) };
        }
      }
      break;
    }

    const timer = new AbortController();
    const timeout = setTimeout(() => timer.abort(new DOMException("timeout", "TimeoutError")), opts.timeoutMs);
    const onOuterAbort = () => timer.abort(new DOMException("aborted", "AbortError"));
    opts.signal?.addEventListener("abort", onOuterAbort, { once: true });

    try {
      const res = await fetch(`${BASE}${path}`, { ...init, signal: timer.signal });

      if (!res.ok) {
        const detail = await res.text().catch(() => undefined);
        last = err(res.status >= 500 ? "server" : "bad_request", attempt + 1, {
          status: res.status,
          detail: detail?.slice(0, 400),
        });
      } else {
        const body = await res.json().catch(() => undefined);
        const value = parse(body);
        if (value === undefined) {
          last = err("malformed", attempt + 1, { detail: JSON.stringify(body)?.slice(0, 400) });
          break; // asking again will not change the shape
        }
        return { ok: true, value };
      }
    } catch (e) {
      if (opts.signal?.aborted) return { ok: false, error: err("cancelled", attempt + 1) };
      const name = e instanceof Error ? e.name : "";
      last = err(name === "TimeoutError" ? "timeout" : "unreachable", attempt + 1, {
        detail: e instanceof Error ? e.message : String(e),
      });
    } finally {
      clearTimeout(timeout);
      opts.signal?.removeEventListener("abort", onOuterAbort);
    }

    if (!last.retryable || attempt === opts.maxAttempts - 1) break;
    opts.onRetry?.(attempt + 1, last);
    try {
      await sleep(backoff(attempt), opts.signal);
    } catch {
      return { ok: false, error: err("cancelled", attempt + 1) };
    }
  }

  return { ok: false, error: last };
}

export function ask(
  question: string,
  opts: { signal?: AbortSignal; onRetry?: RequestOptions["onRetry"] } = {},
): Promise<Result<Answer>> {
  return request(
    "/ask",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question }),
    },
    readAnswer,
    { timeoutMs: ASK_TIMEOUT_MS, maxAttempts: MAX_ATTEMPTS, ...opts },
  );
}

export async function health(signal?: AbortSignal): Promise<HealthReport> {
  const res = await request(
    "/health",
    { method: "GET" },
    (b) => (isRecord(b) ? { model: typeof b.model === "string" ? b.model : undefined } : undefined),
    { timeoutMs: HEALTH_TIMEOUT_MS, maxAttempts: 1, signal },
  );

  return res.ok
    ? { reachable: true, model: res.value.model, checkedAt: Date.now() }
    : { reachable: false, checkedAt: Date.now(), error: res.error };
}
