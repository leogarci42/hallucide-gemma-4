"use client";

import { useCallback, useEffect, useSyncExternalStore } from "react";
import type { Answer, ApiError, Conversation, Turn, TurnState } from "./types";

const KEY = "alien-hallucination.conversations.v1";
const MAX_STORED = 50;

export const uid = () =>
  typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

/** First words of the question, so a thread is recognisable in the list. */
export function titleFrom(question: string): string {
  const t = question.trim().replace(/\s+/g, " ");
  return t.length > 60 ? `${t.slice(0, 57)}...` : t || "Untitled";
}

interface State {
  conversations: Conversation[];
  activeId: string | null;
  hydrated: boolean;
}

const EMPTY: State = { conversations: [], activeId: null, hydrated: false };

let state: State = EMPTY;
const listeners = new Set<() => void>();

/** In-flight requests, keyed by turn, so any of them can be cancelled. */
const inflight = new Map<string, { conversationId: string; controller: AbortController }>();

function set(next: Partial<State>) {
  state = { ...state, ...next };
  persist();
  for (const l of listeners) l();
}

function persist() {
  if (typeof window === "undefined" || !state.hydrated) return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(state.conversations.slice(0, MAX_STORED)));
  } catch {
    // quota or private mode; the session keeps working in memory
  }
}

function readStored(): Conversation[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (c): c is Conversation =>
        typeof c === "object" &&
        c !== null &&
        typeof (c as Conversation).id === "string" &&
        Array.isArray((c as Conversation).turns),
    );
  } catch {
    return []; // corrupt storage must not take the app down
  }
}

/** Called once from an effect. A turn left pending by a reload is not pending
    any more, so it is marked as cancelled rather than spinning forever. */
export function hydrate() {
  if (state.hydrated) return;
  const stored = readStored().map((c) => ({
    ...c,
    turns: c.turns.map((t) =>
      t.state.phase === "pending" ? { ...t, state: { phase: "cancelled" } as TurnState } : t,
    ),
  }));
  state = { conversations: stored, activeId: stored[0]?.id ?? null, hydrated: true };
  persist();
  for (const l of listeners) l();
}

function subscribe(onChange: () => void) {
  listeners.add(onChange);
  return () => listeners.delete(onChange);
}

const getSnapshot = () => state;
const getServerSnapshot = () => EMPTY;

/* ── mutations ─────────────────────────────────────────────────────── */

function patchTurn(conversationId: string, turnId: string, turnState: TurnState) {
  set({
    conversations: state.conversations.map((c) =>
      c.id !== conversationId
        ? c
        : {
            ...c,
            updatedAt: Date.now(),
            turns: c.turns.map((t) => (t.id === turnId ? { ...t, state: turnState } : t)),
          },
    ),
  });
}

function startTurn(question: string, conversationId?: string): { conversationId: string; turnId: string } {
  const now = Date.now();
  const turn: Turn = { id: uid(), question, askedAt: now, state: { phase: "pending" } };
  const target = conversationId ?? state.activeId;
  const exists = target !== null && state.conversations.some((c) => c.id === target);

  if (exists) {
    set({
      activeId: target,
      conversations: state.conversations.map((c) =>
        c.id === target ? { ...c, updatedAt: now, turns: [...c.turns, turn] } : c,
      ),
    });
    return { conversationId: target!, turnId: turn.id };
  }

  const created: Conversation = {
    id: uid(),
    title: titleFrom(question),
    createdAt: now,
    updatedAt: now,
    turns: [turn],
  };
  set({ activeId: created.id, conversations: [created, ...state.conversations] });
  return { conversationId: created.id, turnId: turn.id };
}

function registerInflight(turnId: string, conversationId: string, controller: AbortController) {
  inflight.set(turnId, { conversationId, controller });
}

function cancel(turnId: string) {
  const p = inflight.get(turnId);
  if (!p) return;
  p.controller.abort();
  inflight.delete(turnId);
  patchTurn(p.conversationId, turnId, { phase: "cancelled" });
}

function cancelAll() {
  for (const id of [...inflight.keys()]) cancel(id);
}

function settle(
  conversationId: string,
  turnId: string,
  result: { answer?: Answer; error?: ApiError },
) {
  inflight.delete(turnId);
  patchTurn(
    conversationId,
    turnId,
    result.answer ? { phase: "done", answer: result.answer } : { phase: "error", error: result.error! },
  );
}

function newConversation() {
  cancelAll();
  set({ activeId: null });
}

function select(id: string) {
  set({ activeId: id });
}

function rename(id: string, title: string) {
  const clean = title.trim();
  if (!clean) return;
  set({
    conversations: state.conversations.map((c) => (c.id === id ? { ...c, title: clean } : c)),
  });
}

function remove(id: string) {
  const conv = state.conversations.find((c) => c.id === id);
  conv?.turns.forEach((t) => {
    if (inflight.has(t.id)) cancel(t.id);
  });
  const next = state.conversations.filter((c) => c.id !== id);
  set({ conversations: next, activeId: state.activeId === id ? (next[0]?.id ?? null) : state.activeId });
}

function dropTurn(conversationId: string, turnId: string) {
  if (inflight.has(turnId)) cancel(turnId);
  set({
    conversations: state.conversations.map((c) =>
      c.id === conversationId ? { ...c, turns: c.turns.filter((t) => t.id !== turnId) } : c,
    ),
  });
}

export const actions = {
  startTurn,
  registerInflight,
  patchTurn,
  settle,
  cancel,
  cancelAll,
  newConversation,
  select,
  rename,
  remove,
  dropTurn,
};

export function useConversations() {
  const snapshot = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  useEffect(() => {
    hydrate();
  }, []);

  const active = useCallback(
    () => snapshot.conversations.find((c) => c.id === snapshot.activeId) ?? null,
    [snapshot],
  )();

  return { ...snapshot, active, ...actions };
}
