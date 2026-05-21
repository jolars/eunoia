// Light/dark theme handling. The effective theme is applied as a `dark` class
// on <html>; the `app.css` tokens key off that class. A matching inline script
// in `app.html` applies the same logic before first paint to avoid a flash of
// the wrong theme — keep the storage key and resolution logic in sync.

const STORAGE_KEY = "eunoia.theme";

// `system` is the default (no stored value): the theme follows the OS until
// the user clicks the toggle, which commits an explicit `light`/`dark`. The UI
// only ever flips between light and dark — `system` isn't a button state.
export type ThemeChoice = "system" | "light" | "dark";

function readStored(): ThemeChoice {
  if (typeof localStorage === "undefined") return "system";
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "light" || v === "dark" || v === "system") return v;
  } catch {
    // ignore unavailable / blocked storage
  }
  return "system";
}

function prefersDark(): boolean {
  return (
    typeof matchMedia !== "undefined" &&
    matchMedia("(prefers-color-scheme: dark)").matches
  );
}

class Theme {
  /** The persisted user choice; `system` until the toggle is first clicked. */
  choice: ThemeChoice = $state(readStored());
  /** Live OS preference, kept reactive so the icon updates in `system` mode. */
  systemDark: boolean = $state(prefersDark());

  /** The theme actually in effect (`system` resolved against the OS). */
  get resolved(): "light" | "dark" {
    if (this.choice === "system") return this.systemDark ? "dark" : "light";
    return this.choice;
  }

  /** Toggle the `dark` class on <html> to match the resolved theme. */
  apply() {
    if (typeof document === "undefined") return;
    document.documentElement.classList.toggle("dark", this.resolved === "dark");
  }

  set(choice: ThemeChoice) {
    this.choice = choice;
    if (typeof localStorage !== "undefined") {
      try {
        localStorage.setItem(STORAGE_KEY, choice);
      } catch {
        // ignore quota / blocked storage
      }
    }
    this.apply();
  }

  /** Flip to the opposite of the current theme (commits an explicit choice). */
  toggle() {
    this.set(this.resolved === "dark" ? "light" : "dark");
  }
}

export const theme = new Theme();

/**
 * Apply the stored theme and keep `system` mode tracking the OS live. Call once
 * on mount; returns a cleanup function that detaches the media listener.
 */
export function initTheme(): () => void {
  theme.apply();
  if (typeof matchMedia === "undefined") return () => {};
  const mq = matchMedia("(prefers-color-scheme: dark)");
  const onChange = () => {
    theme.systemDark = mq.matches;
    if (theme.choice === "system") theme.apply();
  };
  mq.addEventListener("change", onChange);
  return () => mq.removeEventListener("change", onChange);
}
