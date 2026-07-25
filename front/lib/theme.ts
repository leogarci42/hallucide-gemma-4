"use client";

import { useCallback, useSyncExternalStore } from "react";

export type Theme = "light" | "dark";

const KEY = "alien-hallucination.theme";
const EVENT = "alien-hallucination:themechange";

/** Runs before paint so the first frame is already in the right theme.
    ?theme=light|dark wins, which makes the demo deep-linkable. */
export const THEME_BOOT_SCRIPT = `(function(){try{
var q=new URLSearchParams(location.search).get('theme');
var s=q||localStorage.getItem(${JSON.stringify(KEY)});
var t=s==='light'||s==='dark'?s:(matchMedia('(prefers-color-scheme: light)').matches?'light':'dark');
document.documentElement.dataset.theme=t;
}catch(e){document.documentElement.dataset.theme='dark';}})();`;

function subscribe(onChange: () => void) {
  window.addEventListener(EVENT, onChange);
  return () => window.removeEventListener(EVENT, onChange);
}

/** The DOM is the source of truth: the boot script set it before React ran. */
function getSnapshot(): Theme {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

const getServerSnapshot = (): Theme => "dark";

export function useTheme() {
  const theme = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const setTheme = useCallback((next: Theme) => {
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(KEY, next);
    } catch {
      // private mode; the choice lasts for this session only
    }
    window.dispatchEvent(new Event(EVENT));
  }, []);

  const toggle = useCallback(
    () => setTheme(getSnapshot() === "light" ? "dark" : "light"),
    [setTheme],
  );

  return { theme, setTheme, toggle };
}
