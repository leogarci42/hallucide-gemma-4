import { NextResponse } from "next/server";
import { ENGINE_ORIGIN, engineFetch } from "../engine";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

const ASK_TIMEOUT_MS = 90_000;
const MAX_QUESTION_CHARS = 600;

const noStore = { "cache-control": "no-store" };

export async function POST(request: Request) {
  let question: unknown;
  try {
    const body = await request.json();
    question = (body as { question?: unknown })?.question;
  } catch {
    return NextResponse.json({ error: "body must be JSON" }, { status: 400, headers: noStore });
  }

  if (typeof question !== "string" || !question.trim()) {
    return NextResponse.json({ error: "question is required" }, { status: 400, headers: noStore });
  }

  if (question.length > MAX_QUESTION_CHARS) {
    return NextResponse.json(
      { error: `question must be at most ${MAX_QUESTION_CHARS} characters` },
      { status: 400, headers: noStore },
    );
  }

  if (!ENGINE_ORIGIN) {
    return NextResponse.json(
      { error: "ENGINE_ORIGIN is not set" },
      { status: 503, headers: noStore },
    );
  }

  const res = await engineFetch(
    "/ask",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question: question.trim() }),
    },
    ASK_TIMEOUT_MS,
  );

  if (!res.ok) {
    // 5xx so the client treats it as worth retrying
    return NextResponse.json({ error: res.error }, { status: res.status ?? 502, headers: noStore });
  }

  return NextResponse.json(res.body, { headers: noStore });
}
