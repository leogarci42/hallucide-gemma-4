"use client";

import { useEffect, useState } from "react";
import type { ApiError, Turn as TurnData, Verdict } from "@/lib/types";
import styles from "./turn.module.css";

const VERDICT_LABEL: Record<Verdict, string> = {
  grounded: "Every claim cleared both checks",
  partial: "Some claims did not clear both checks",
  unsupported: "No claim cleared both checks",
  refused: "Declined: no dataset covers this",
};

const CLAIM_NOTE: Record<string, string> = {
  hallucinated: "Withheld. It does not hold against the passage.",
  unverifiable: "Withheld. Neither check could settle it.",
};

const CLAIM_CLASS: Record<string, string> = {
  grounded: styles.claimOk,
  hallucinated: styles.claimKo,
  unverifiable: styles.claimUnknown,
};

const VERDICT_TONE: Record<Verdict, string> = {
  grounded: styles.toneGrounded,
  partial: styles.tonePartial,
  unsupported: styles.toneUnsupported,
  refused: styles.toneRefused,
};

/* A local model on a laptop can take a while. Showing the clock move is the
   difference between "thinking" and "frozen" when someone is watching. */
function Waiting({ since }: { since: number }) {
  const [elapsed, setElapsed] = useState(() => Math.floor((Date.now() - since) / 1000));

  useEffect(() => {
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - since) / 1000)), 1000);
    return () => clearInterval(t);
  }, [since]);

  return (
    <span className={styles.working}>
      <span className={styles.pulse} aria-hidden />
      Checking each claim against the sources
      {elapsed > 2 && <span className={styles.elapsed}>{elapsed}s</span>}
    </span>
  );
}

function ErrorNotice({
  error,
  onRetry,
  onDismiss,
}: {
  error: ApiError;
  onRetry: () => void;
  onDismiss: () => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`${styles.panel} ${styles.error}`}>
      <p className={styles.errorHead}>{error.message}</p>
      {error.attempts > 1 && (
        <p className={styles.errorMeta}>Tried {error.attempts} times.</p>
      )}
      <div className={styles.actions}>
        {error.retryable && (
          <button className={styles.action} onClick={onRetry}>
            Try again
          </button>
        )}
        <button className={styles.actionGhost} onClick={onDismiss}>
          Remove
        </button>
        {error.detail && (
          <button className={styles.actionGhost} onClick={() => setOpen((v) => !v)}>
            {open ? "Hide details" : "Details"}
          </button>
        )}
      </div>
      {open && error.detail && <pre className={styles.detail}>{error.detail}</pre>}
    </div>
  );
}

/* Copying the checked answer is the point of the product: what leaves the app
   is the version with the unsupported sentences already removed. */
function CopyButton({ text }: { text: string }) {
  const [state, setState] = useState<"idle" | "done" | "failed">("idle");

  useEffect(() => {
    if (state === "idle") return;
    const t = setTimeout(() => setState("idle"), 2000);
    return () => clearTimeout(t);
  }, [state]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setState("done");
    } catch {
      setState("failed");
    }
  }

  return (
    <button className={styles.copy} onClick={copy} disabled={!text}>
      {state === "done" ? "Copied" : state === "failed" ? "Copy blocked" : "Copy"}
    </button>
  );
}

export default function Turn({
  turn,
  onCancel,
  onRetry,
  onDrop,
}: {
  turn: TurnData;
  onCancel: () => void;
  onRetry: () => void;
  onDrop: () => void;
}) {
  const [view, setView] = useState<"verified" | "draft">("verified");
  const s = turn.state;

  /* Sources the engine returned that no claim ended up citing. Worth showing:
     they are what was read, even where nothing survived the check. */
  const extraSources =
    s.phase === "done"
      ? s.answer.sources.filter(
          (src) => !s.answer.claims.some((c) => c.source?.id === src.id),
        )
      : [];

  return (
    <article className={styles.turn}>
      <p className={styles.role}>You asked</p>
      <h2 className={styles.question}>{turn.question}</h2>

      {s.phase === "pending" && (
        <div className={styles.panel}>
          <Waiting since={turn.askedAt} />
          <div className={styles.actions}>
            <button className={styles.actionGhost} onClick={onCancel}>
              Cancel
            </button>
          </div>
        </div>
      )}

      {s.phase === "cancelled" && (
        <div className={styles.panel}>
          <p className={styles.quiet}>Cancelled before an answer came back.</p>
          <div className={styles.actions}>
            <button className={styles.action} onClick={onRetry}>
              Ask again
            </button>
            <button className={styles.actionGhost} onClick={onDrop}>
              Remove
            </button>
          </div>
        </div>
      )}

      {s.phase === "error" && (
        <ErrorNotice error={s.error} onRetry={onRetry} onDismiss={onDrop} />
      )}

      {s.phase === "done" && (
        <div className={styles.panel}>
          <header className={styles.answerHead}>
            <span className={`${styles.verdict} ${VERDICT_TONE[s.answer.verdict]}`}>
              {VERDICT_LABEL[s.answer.verdict]}
            </span>

            <div className={styles.segmented} role="group" aria-label="Answer view">
              <button
                className={view === "verified" ? styles.segOn : styles.seg}
                aria-pressed={view === "verified"}
                onClick={() => setView("verified")}
              >
                After checking
              </button>
              <button
                className={view === "draft" ? styles.segOn : styles.seg}
                aria-pressed={view === "draft"}
                onClick={() => setView("draft")}
              >
                Model draft
              </button>
            </div>
          </header>

          {view === "draft" ? (
            <div className={styles.body}>
              <p className={styles.quiet}>
                What Gemma wrote before anything was checked.
              </p>
              <p className={styles.draft}>{s.answer.draft}</p>
            </div>
          ) : (
            <div className={styles.body}>
              {s.answer.verified && <p className={styles.verified}>{s.answer.verified}</p>}

              {s.answer.claims.length > 0 ? (
                <>
                <p className={styles.sectionLabel}>
                  {s.answer.claims.length === 1 ? "The claim behind it" : "The claims behind it"}
                </p>
                <ul className={styles.claims}>
                  {s.answer.claims.map((c) => (
                    <li key={c.id} className={CLAIM_CLASS[c.status]}>
                      <span className={styles.claimMark} aria-hidden />
                      <div>
                        <p className={styles.claimText}>{c.text}</p>
                        {c.status === "grounded" && c.source ? (
                          <p className={styles.claimSource}>
                            {c.source.url ? (
                              <a href={c.source.url} target="_blank" rel="noreferrer noopener">
                                {c.source.title}
                              </a>
                            ) : (
                              c.source.title
                            )}
                          </p>
                        ) : (
                          <p className={styles.claimSource}>{CLAIM_NOTE[c.status]}</p>
                        )}
                        {(c.semanticPass !== undefined || c.literalPass !== undefined) && (
                          <p className={styles.lanes}>
                            <span>
                              semantic {c.semanticPass === undefined ? "n/a" : c.semanticPass ? "pass" : "fail"}
                            </span>
                            <span>
                              figures {c.literalPass === undefined ? "n/a" : c.literalPass ? "pass" : "fail"}
                            </span>
                          </p>
                        )}
                        {c.source?.passage && (
                          <blockquote className={styles.passage}>{c.source.passage}</blockquote>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
                </>
              ) : (
                <p className={styles.quiet}>
                  The engine returned no claim it could stand behind.
                </p>
              )}
            </div>
          )}

          {extraSources.length > 0 && view === "verified" && (
            <div className={styles.sources}>
              <p className={styles.sectionLabel}>Also read</p>
              <ul>
                {extraSources.map((src) => (
                  <li key={src.id}>
                    {src.url ? (
                      <a href={src.url} target="_blank" rel="noreferrer noopener">
                        {src.title}
                      </a>
                    ) : (
                      src.title
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <footer className={styles.meta}>
            {s.answer.model && <span>{s.answer.model}</span>}
            {s.answer.dataset && <span>{s.answer.dataset}</span>}
            {s.answer.contextPassages !== undefined && (
              <span>{s.answer.contextPassages} passages injected</span>
            )}
            {s.answer.contextChars !== undefined && (
              <span>{s.answer.contextChars.toLocaleString("en-US")} context characters</span>
            )}
            {s.answer.contextTokens !== undefined && (
              <span>{s.answer.contextTokens.toLocaleString("en-US")} context tokens</span>
            )}
            {s.answer.latencyMs !== undefined && (
              <span>{(s.answer.latencyMs / 1000).toFixed(1)}s</span>
            )}
            <CopyButton text={view === "draft" ? s.answer.draft : s.answer.verified || s.answer.draft} />
          </footer>
        </div>
      )}
    </article>
  );
}
