import { type ComponentProps, mount as mountSvelte, unmount } from "svelte";
import type DiagramExample from "#lib/components/DiagramExample.svelte";
import type PipelinePlayer from "#lib/components/PipelinePlayer.svelte";
import { hydrateFromStorage } from "#lib/state.svelte.ts";
import { observeTheme } from "#lib/theme.svelte.ts";
import "../app.css";

interface Context {
  signal: AbortSignal;
  props:
    | { component: "hero" | "builder" }
    | { component: "diagram"; props: ComponentProps<typeof DiagramExample> }
    | { component: "pipeline"; props: ComponentProps<typeof PipelinePlayer> };
}

export async function mount(element: HTMLElement, { signal, props }: Context) {
  let create: () => ReturnType<typeof mountSvelte>;
  if (props.component === "hero") {
    const { default: Hero } = await import("#lib/components/HeroWidget.svelte");
    create = () => mountSvelte(Hero, { target: element });
  } else if (props.component === "builder") {
    const { default: Builder } = await import(
      "#lib/components/DiagramBuilder.svelte"
    );
    hydrateFromStorage();
    create = () => mountSvelte(Builder, { target: element });
  } else if (props.component === "pipeline") {
    const { default: Pipeline } = await import(
      "#lib/components/PipelinePlayer.svelte"
    );
    create = () =>
      mountSvelte(Pipeline, {
        target: element,
        props: props.props,
      });
  } else if (props.component === "diagram") {
    const { default: Diagram } = await import(
      "#lib/components/DiagramExample.svelte"
    );
    create = () =>
      mountSvelte(Diagram, {
        target: element,
        props: props.props,
      });
  } else {
    throw new Error("Unknown Eunoia widget");
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
