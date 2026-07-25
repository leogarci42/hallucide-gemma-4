"use client";

import "./globals.css";

/* Catches errors thrown in the root layout itself, where the normal error
   boundary cannot render. Ships its own html/body, hence the plain markup. */
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en" data-theme="dark">
      <body>
        <div
          style={{
            height: "100vh",
            display: "grid",
            placeItems: "center",
            padding: "2.4rem",
          }}
        >
          <div
            style={{
              maxWidth: "52rem",
              border: "1px solid var(--line)",
              borderRadius: 3,
              padding: "3.2rem",
            }}
          >
            <h1 style={{ fontSize: "2rem", fontWeight: 400, marginBottom: "0.8rem" }}>
              The app failed to start.
            </h1>
            <p style={{ fontSize: "1.4rem", color: "var(--muted)" }}>{error.message}</p>
            <button
              onClick={reset}
              style={{
                marginTop: "2.4rem",
                background: "none",
                border: "1px solid var(--line)",
                borderRadius: 3,
                fontSize: "1.3rem",
                padding: "1rem 1.8rem",
              }}
            >
              Try again
            </button>
          </div>
        </div>
      </body>
    </html>
  );
}
