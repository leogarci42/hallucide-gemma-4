/* The only place that knows where the verification engine lives.
   Set ENGINE_ORIGIN (server-side, not NEXT_PUBLIC) to point the app at it.
   With nothing set, every route answers 503 and the UI says the engine is
   unreachable. It never invents an answer. */

export const ENGINE_ORIGIN = process.env.ENGINE_ORIGIN?.replace(/\/$/, "") ?? "";

export type EngineResult =
  | { ok: true; body: unknown }
  | { ok: false; error: string; status?: number };

export async function engineFetch(
  path: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<EngineResult> {
  if (!ENGINE_ORIGIN) return { ok: false, error: "ENGINE_ORIGIN is not set" };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${ENGINE_ORIGIN}${path}`, {
      ...init,
      signal: controller.signal,
      cache: "no-store",
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      return { ok: false, error: detail.slice(0, 400) || `engine returned ${res.status}`, status: res.status };
    }

    return { ok: true, body: await res.json() };
  } catch (e) {
    const aborted = e instanceof Error && e.name === "AbortError";
    return { ok: false, error: aborted ? `engine did not answer within ${timeoutMs}ms` : String(e) };
  } finally {
    clearTimeout(timer);
  }
}
