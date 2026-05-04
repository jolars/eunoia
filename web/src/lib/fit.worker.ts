/// <reference lib="webworker" />

import type { FitResult } from "../types/diagram";
import { type FitInputs, runFit } from "./fit";

declare const self: DedicatedWorkerGlobalScope;

type Request =
  | { id: number; type: "init" }
  | { id: number; type: "fit"; inputs: FitInputs };

type Response =
  | { id: number; ready: true }
  | { id: number; result: FitResult | null }
  | { id: number; error: string };

let wasmPromise: Promise<unknown> | null = null;

async function ensureWasm() {
  if (!wasmPromise) {
    wasmPromise = import("@jolars/eunoia");
  }
  return wasmPromise;
}

self.onmessage = async (e: MessageEvent<Request>) => {
  const msg = e.data;
  try {
    await ensureWasm();
    if (msg.type === "init") {
      const reply: Response = { id: msg.id, ready: true };
      self.postMessage(reply);
      return;
    }
    const result = runFit(msg.inputs);
    const reply: Response = { id: msg.id, result };
    self.postMessage(reply);
  } catch (err) {
    const reply: Response = {
      id: msg.id,
      error: err instanceof Error ? err.message : String(err),
    };
    self.postMessage(reply);
  }
};
