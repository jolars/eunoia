/// <reference lib="webworker" />

import { runFit, type FitInputs } from "./fit";
import type { FitResult } from "../types/diagram";

declare const self: DedicatedWorkerGlobalScope;

type Request =
  | { id: number; type: "init" }
  | { id: number; type: "fit"; inputs: FitInputs };

type Response =
  | { id: number; ready: true }
  | { id: number; result: FitResult | null }
  | { id: number; error: string };

let wasmPromise: Promise<unknown> | null = null;

async function loadWasm() {
  if (!wasmPromise) {
    wasmPromise = (async () => {
      const wasm = await import("../../pkg/eunoia_wasm.js");
      await (wasm as { default: () => Promise<unknown> }).default();
      return wasm;
    })();
  }
  return wasmPromise;
}

self.onmessage = async (e: MessageEvent<Request>) => {
  const msg = e.data;
  try {
    const wasm = await loadWasm();
    if (msg.type === "init") {
      const reply: Response = { id: msg.id, ready: true };
      self.postMessage(reply);
      return;
    }
    const result = runFit(wasm, msg.inputs);
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
