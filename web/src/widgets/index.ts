import { type ComponentProps, mount as mountSvelte, unmount } from "svelte";
import { hydrateFromStorage } from "#lib/state.svelte.ts";
import { observeTheme } from "#lib/theme.svelte.ts";
import examples from "./examples.json";
import "../app.css";

interface Context {
  signal: AbortSignal;
  assetBaseUrl: string;
}

export async function mount(element: HTMLElement, { signal }: Context) {
  // The authored block ID selects an example while all instances share one
  // retained bundle and the same cached WASM and worker resources.
  const id = element.parentElement?.id;
  let create: () => ReturnType<typeof mountSvelte>;
  if (id === "home-diagram") {
    const { default: Hero } = await import("#lib/components/HeroWidget.svelte");
    create = () => mountSvelte(Hero, { target: element });
  } else if (id === "diagram-builder") {
    const { default: Builder } = await import(
      "#lib/components/DiagramBuilder.svelte"
    );
    hydrateFromStorage();
    create = () => mountSvelte(Builder, { target: element });
  } else {
    const example = examples[id as keyof typeof examples];
    if (!example) throw new Error(`Unknown Eunoia example: ${id}`);
    if (example.component === "pipeline") {
      const { default: Pipeline } = await import(
        "#lib/components/PipelinePlayer.svelte"
      );
      create = () =>
        mountSvelte(Pipeline, {
          target: element,
          props: example.props as ComponentProps<typeof Pipeline>,
        });
    } else {
      const { default: Diagram } = await import(
        "#lib/components/DiagramExample.svelte"
      );
      create = () =>
        mountSvelte(Diagram, {
          target: element,
          props: example.props as ComponentProps<typeof Diagram>,
        });
    }
  }
  if (signal.aborted) return;
  element.classList.add("eunoia-widget");
  const stopTheme = observeTheme(element);
  try {
    const app = create();
    return () => {
      stopTheme();
      void unmount(app);
    };
  } catch (error) {
    stopTheme();
    throw error;
  }
}
