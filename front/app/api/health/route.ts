import { NextResponse } from "next/server";
import { ENGINE_ORIGIN, engineFetch } from "../engine";

export const dynamic = "force-dynamic";

export async function GET() {
  if (!ENGINE_ORIGIN) {
    return NextResponse.json(
      { error: "ENGINE_ORIGIN is not set" },
      { status: 503, headers: { "cache-control": "no-store" } },
    );
  }

  const res = await engineFetch("/health", { method: "GET" }, 4_000);

  if (!res.ok) {
    return NextResponse.json({ error: res.error }, { status: 503, headers: { "cache-control": "no-store" } });
  }

  const model =
    typeof res.body === "object" && res.body !== null && "model" in res.body
      ? (res.body as { model?: unknown }).model
      : undefined;

  return NextResponse.json(
    { model: typeof model === "string" ? model : undefined },
    { headers: { "cache-control": "no-store" } },
  );
}
