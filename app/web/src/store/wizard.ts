import { create } from "zustand";
import { persist } from "zustand/middleware";

interface WizardSlice {
  /** Per wizard mode: prompt + last AI-fill response + warnings */
  prompts: Record<string, string>;
  payloads: Record<string, Record<string, unknown> | undefined>;
  warnings: Record<string, string[] | undefined>;
  pasteBlocks: Record<string, Array<{ label?: string; content: string }> | undefined>;
  /** Last successful generation result per mode (audio + metadata) */
  lastResult: Record<string, Record<string, unknown> | undefined>;

  setPrompt: (mode: string, value: string) => void;
  setHydration: (
    mode: string,
    data: {
      payload?: Record<string, unknown>;
      warnings?: string[];
      paste_blocks?: Array<{ label?: string; content: string }>;
    },
  ) => void;
  setResult: (mode: string, result: Record<string, unknown>) => void;
  reset: (mode: string) => void;
}

export const useWizardStore = create<WizardSlice>()(
  persist(
    (set) => ({
      prompts: {},
      payloads: {},
      warnings: {},
      pasteBlocks: {},
      lastResult: {},
      setPrompt: (mode, value) =>
        set((s) => ({ prompts: { ...s.prompts, [mode]: value } })),
      setHydration: (mode, data) =>
        set((s) => ({
          payloads: { ...s.payloads, [mode]: data.payload },
          warnings: { ...s.warnings, [mode]: data.warnings },
          pasteBlocks: { ...s.pasteBlocks, [mode]: data.paste_blocks },
        })),
      setResult: (mode, result) =>
        set((s) => ({ lastResult: { ...s.lastResult, [mode]: result } })),
      reset: (mode) =>
        set((s) => ({
          prompts: { ...s.prompts, [mode]: "" },
          payloads: { ...s.payloads, [mode]: undefined },
          warnings: { ...s.warnings, [mode]: undefined },
          pasteBlocks: { ...s.pasteBlocks, [mode]: undefined },
          lastResult: { ...s.lastResult, [mode]: undefined },
        })),
    }),
    {
      name: "acejam-wizard-state",
      partialize: (state) => ({
        prompts: state.prompts,
        payloads: state.payloads,
        warnings: state.warnings,
        pasteBlocks: state.pasteBlocks,
      }),
    },
  ),
);
