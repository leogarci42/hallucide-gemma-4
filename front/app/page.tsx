"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import Composer from "@/components/Composer";
import Flow from "@/components/Flow";
import Sidebar from "@/components/Sidebar";
import TurnView from "@/components/Turn";
import { ask, health } from "@/lib/api";
import { actions, useConversations } from "@/lib/store";
import { useTheme } from "@/lib/theme";
import type { HealthReport } from "@/lib/types";
import styles from "./page.module.css";

const HEALTH_INTERVAL_MS = 30_000;

/* navigator.onLine is external state, so it is read rather than mirrored. */
function subscribeOnline(onChange: () => void) {
  window.addEventListener("online", onChange);
  window.addEventListener("offline", onChange);
  return () => {
    window.removeEventListener("online", onChange);
    window.removeEventListener("offline", onChange);
  };
}

export default function Page() {
  const store = useConversations();
  const { theme, toggle } = useTheme();
  const online = useSyncExternalStore(
    subscribeOnline,
    () => navigator.onLine,
    () => true,
  );

  const [engine, setEngine] = useState<HealthReport | null>(null);
  const [retryNotice, setRetryNotice] = useState<string | null>(null);
  // below 860px the sidebar is a drawer rather than a column
  const [drawer, setDrawer] = useState(false);
  const stream = useRef<HTMLDivElement>(null);

  /* ── engine health, polled gently and refreshed when the tab wakes up ── */
  useEffect(() => {
    let alive = true;
    let controller: AbortController | null = null;

    async function check() {
      if (document.hidden) return;
      controller?.abort();
      controller = new AbortController();
      const report = await health(controller.signal);
      if (alive) setEngine(report);
    }

    check();
    const timer = setInterval(check, HEALTH_INTERVAL_MS);
    document.addEventListener("visibilitychange", check);
    window.addEventListener("online", check);

    return () => {
      alive = false;
      controller?.abort();
      clearInterval(timer);
      document.removeEventListener("visibilitychange", check);
      window.removeEventListener("online", check);
    };
  }, []);

  const active = store.active;
  const turns = active?.turns ?? [];
  const busyTurn = turns.find((t) => t.state.phase === "pending") ?? null;

  useEffect(() => {
    stream.current?.scrollTo({ top: stream.current.scrollHeight, behavior: "smooth" });
  }, [turns.length, busyTurn?.id]);

  /* ── asking ── */
  const run = useCallback(
    async (question: string, conversationId?: string, existingTurnId?: string) => {
      let target: { conversationId: string; turnId: string };

      if (existingTurnId && conversationId) {
        target = { conversationId, turnId: existingTurnId };
        actions.patchTurn(conversationId, existingTurnId, { phase: "pending" });
      } else {
        target = actions.startTurn(question, conversationId);
      }

      const controller = new AbortController();
      actions.registerInflight(target.turnId, target.conversationId, controller);

      const result = await ask(question, {
        signal: controller.signal,
        onRetry: (attempt) => setRetryNotice(`No answer yet. Attempt ${attempt + 1}.`),
      });

      setRetryNotice(null);

      if (result.ok) {
        actions.settle(target.conversationId, target.turnId, { answer: result.value });
      } else if (result.error.kind !== "cancelled") {
        actions.settle(target.conversationId, target.turnId, { error: result.error });
      }
    },
    [],
  );

  const submit = useCallback((q: string) => void run(q), [run]);

  const retry = useCallback(
    (turnId: string, question: string) => {
      if (!active) return;
      void run(question, active.id, turnId);
    },
    [active, run],
  );

  /* ── shortcuts ── */
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        actions.newConversation();
      }
      if (e.key === "Escape") {
        if (drawer) setDrawer(false);
        else if (busyTurn) actions.cancel(busyTurn.id);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busyTurn, drawer]);

  /* The model name is printed only when the engine itself reported one.
     Nothing here is a placeholder. */
  const engineLabel = !online
    ? "No network"
    : engine === null
      ? "Checking engine"
      : engine.reachable
        ? (engine.model ?? "Engine ready")
        : "Engine unreachable";

  const engineOnline = !online ? false : engine === null ? null : engine.reachable;

  return (
    <>
      <div className={styles.frame} aria-hidden />

      {drawer && (
        <button
          className={styles.scrim}
          onClick={() => setDrawer(false)}
          aria-label="Close conversations"
        />
      )}

      <div className={styles.app}>
        <Sidebar
          conversations={store.conversations}
          activeId={store.activeId}
          hydrated={store.hydrated}
          theme={theme}
          open={drawer}
          onClose={() => setDrawer(false)}
          onNew={actions.newConversation}
          onSelect={actions.select}
          onRename={actions.rename}
          onRemove={actions.remove}
          onToggleTheme={toggle}
        />

        <section className={styles.chat}>
          <header className={styles.head}>
            <button
              className={styles.menu}
              onClick={() => setDrawer(true)}
              aria-label="Show conversations"
            >
              <span aria-hidden>Menu</span>
            </button>
            <h1>{active ? active.title : "New question"}</h1>
            {retryNotice && <span className={styles.notice}>{retryNotice}</span>}
            <p
              className={
                engineOnline === null
                  ? styles.statusIdle
                  : engineOnline
                    ? styles.statusOn
                    : styles.statusOff
              }
              role="status"
            >
              <span className={styles.statusDot} aria-hidden />
              {engineLabel}
            </p>
          </header>

          <div className={styles.body}>
            <div
              className={turns.length === 0 ? styles.stream : `${styles.stream} ${styles.streamScroll}`}
              ref={stream}
              aria-live="polite"
              aria-busy={Boolean(busyTurn)}
            >
              {turns.length === 0 ? (
                <Flow />
              ) : (
                <div className={styles.turns}>
                  {turns.map((t) => (
                    <TurnView
                      key={t.id}
                      turn={t}
                      onCancel={() => actions.cancel(t.id)}
                      onRetry={() => retry(t.id, t.question)}
                      onDrop={() => active && actions.dropTurn(active.id, t.id)}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>

          <Composer
            busy={Boolean(busyTurn)}
            onSubmit={submit}
            onCancel={() => busyTurn && actions.cancel(busyTurn.id)}
          />
        </section>
      </div>
    </>
  );
}
