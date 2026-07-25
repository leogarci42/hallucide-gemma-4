"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./composer.module.css";

const MAX_CHARS = 600;

export default function Composer({
  busy,
  onSubmit,
  onCancel,
}: {
  busy: boolean;
  onSubmit: (question: string) => void;
  onCancel: () => void;
}) {
  const [value, setValue] = useState("");
  const box = useRef<HTMLTextAreaElement>(null);

  // grow with the text, up to a ceiling, so long questions stay readable
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  useEffect(() => {
    if (!busy) box.current?.focus();
  }, [busy]);

  function send() {
    const q = value.trim();
    if (!q || busy) return;
    onSubmit(q);
    setValue("");
  }

  return (
    <form
      className={styles.composer}
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
    >
      <div className={styles.field}>
        <textarea
          ref={box}
          rows={1}
          value={value}
          maxLength={MAX_CHARS}
          placeholder="Ask a question"
          aria-label="Your question"
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        {value.length > MAX_CHARS - 100 && (
          <span className={styles.count}>{MAX_CHARS - value.length}</span>
        )}
      </div>

      {busy ? (
        <button type="button" className={styles.cancel} onClick={onCancel}>
          Cancel
        </button>
      ) : (
        <button type="submit" className={styles.send} disabled={!value.trim()}>
          Ask
        </button>
      )}
    </form>
  );
}
