"use client";

import { useEffect, useRef, useState } from "react";
import type { Conversation } from "@/lib/types";
import Logo from "./Logo";
import styles from "./sidebar.module.css";

function relative(ts: number): string {
  const mins = Math.round((Date.now() - ts) / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

function Row({
  conversation,
  active,
  onSelect,
  onRename,
  onRemove,
}: {
  conversation: Conversation;
  active: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onRemove: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(conversation.title);
  const [confirming, setConfirming] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) input.current?.select();
  }, [editing]);

  useEffect(() => {
    if (!confirming) return;
    const t = setTimeout(() => setConfirming(false), 4000);
    return () => clearTimeout(t);
  }, [confirming]);

  function commit() {
    onRename(draft);
    setEditing(false);
  }

  if (editing) {
    return (
      <div className={styles.rowEdit}>
        <input
          ref={input}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") {
              setDraft(conversation.title);
              setEditing(false);
            }
          }}
          aria-label="Conversation name"
        />
      </div>
    );
  }

  return (
    <div className={active ? styles.rowOn : styles.row}>
      <button className={styles.rowMain} onClick={onSelect} title={conversation.title}>
        <span className={styles.rowTitle}>{conversation.title}</span>
        <span className={styles.rowMeta}>
          {conversation.turns.length} {conversation.turns.length === 1 ? "question" : "questions"}
          {" · "}
          {relative(conversation.updatedAt)}
        </span>
      </button>

      <div className={styles.rowTools}>
        <button
          className={styles.tool}
          onClick={() => {
            setDraft(conversation.title);
            setEditing(true);
          }}
          aria-label={`Rename ${conversation.title}`}
          title="Rename"
        >
          Rename
        </button>
        <button
          className={confirming ? styles.toolWarn : styles.tool}
          onClick={() => (confirming ? onRemove() : setConfirming(true))}
          aria-label={`Delete ${conversation.title}`}
          title={confirming ? "Click again to delete" : "Delete"}
        >
          {confirming ? "Sure?" : "Delete"}
        </button>
      </div>
    </div>
  );
}

export default function Sidebar({
  conversations,
  activeId,
  hydrated,
  theme,
  open,
  onNew,
  onSelect,
  onRename,
  onRemove,
  onToggleTheme,
  onClose,
}: {
  conversations: Conversation[];
  activeId: string | null;
  hydrated: boolean;
  theme: "light" | "dark";
  /** Only meaningful on narrow screens, where the sidebar is a drawer. */
  open: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onRemove: (id: string) => void;
  onToggleTheme: () => void;
  onClose: () => void;
}) {
  return (
    <aside className={open ? `${styles.side} ${styles.open}` : styles.side}>
      <div className={styles.brand}>
        <Logo height={34} />
      </div>

      <div className={styles.body}>
        <button
          className={styles.new}
          onClick={() => {
            onNew();
            onClose();
          }}
        >
          New question
        </button>

        <p className={styles.label}>History</p>

        <div className={styles.list}>
          {!hydrated ? null : conversations.length === 0 ? (
            <p className={styles.empty}>Nothing yet.</p>
          ) : (
            conversations.map((c) => (
              <Row
                key={c.id}
                conversation={c}
                active={c.id === activeId}
                onSelect={() => {
                  onSelect(c.id);
                  onClose();
                }}
                onRename={(t) => onRename(c.id, t)}
                onRemove={() => onRemove(c.id)}
              />
            ))
          )}
        </div>
      </div>

      <div className={styles.foot}>
        <button className={styles.themeBtn} onClick={onToggleTheme}>
          {theme === "dark" ? "Light mode" : "Dark mode"}
        </button>
      </div>
    </aside>
  );
}
