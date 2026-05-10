/// <reference lib="webworker" />

import type { FitInputs } from "./fit";
import type { FitResult } from "./types/diagram";

declare const self: DedicatedWorkerGlobalScope;

type Request =
  | { id: number; type: "init" }
  | { id: number; type: "fit"; inputs: FitInputs };

type Response =
  | { id: number; ready: true }
  | { id: number; result: FitResult | null }
  | { id: number; error: string };

// Imports must be type-only at the top level: any value import that
// transitively loads `@jolars/eunoia` pulls in wasm-bindgen's top-level await,
// which `vite-plugin-top-level-await` wraps the entire worker body in. That
// would defer the `self.onmessage = …` assignment until after wasm finishes
// loading — by which time the main thread's first `postMessage` has already
// been dispatched to a worker with no handler and silently dropped.
let runtimePromise: Promise<typeof import("./fit")> | null = null;
async function getRuntime() {
  if (!runtimePromise) runtimePromise = import("./fit");
  return runtimePromise;
}

self.onmessage = async (e: MessageEvent<Request>) => {
  const msg = e.data;
  try {
    const { runFit } = await getRuntime();
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
