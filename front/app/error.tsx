"use client";

import { useEffect } from "react";
import styles from "./error.module.css";

/* Last line of defence: a render error in the chat must not leave a blank
   screen on stage. The conversation history lives in localStorage, so
   reloading brings it back. */
export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("render error", error);
  }, [error]);

  return (
    <div className={styles.wrap}>
      <div className={styles.card}>
        <h1>Something broke in the interface.</h1>
        <p>Your conversations are saved. Reloading brings them back.</p>
        <div className={styles.actions}>
          <button onClick={reset}>Try again</button>
          <button onClick={() => location.reload()}>Reload</button>
        </div>
        {error.message && <pre className={styles.detail}>{error.message}</pre>}
      </div>
    </div>
  );
}
