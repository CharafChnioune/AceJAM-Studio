import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface PasteBlock {
  label?: string;
  content: string;
}

export function normalizeWarnings(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item ?? "").trim()).filter(Boolean);
  }
  if (typeof value === "string" && value.trim()) return [value.trim()];
  return [];
}

export function normalizePasteBlocks(value: unknown): PasteBlock[] {
  if (Array.isArray(value)) {
    return value
      .map((item, index): PasteBlock | undefined => {
        if (typeof item === "string") {
          const content = item.trim();
          return content ? { label: `Block ${index + 1}`, content } : undefined;
        }
        if (item && typeof item === "object") {
          const record = item as Record<string, unknown>;
          const content = String(record.content ?? record.text ?? "").trim();
          if (!content) return undefined;
          const label = String(
            record.label ?? record.title ?? `Block ${index + 1}`,
          ).trim();
          return { label, content };
        }
        return undefined;
      })
      .filter((item): item is PasteBlock => Boolean(item));
  }
  if (typeof value === "string" && value.trim()) {
    return [{ label: "Paste block", content: value.trim() }];
  }
  return [];
}

interface WizardSlice {
  /** Per wizard mode: prompt + last AI-fill response + warnings */
  prompts: Record<string, string>;
  payloads: Record<string, Record<string, unknown> | undefined>;
  warnings: Record<string, string[] | undefined>;
  pasteBlocks: Record<string, PasteBlock[] | undefined>;
  /** Persisted form values per wizard mode. */
  drafts: Record<string, Record<string, unknown> | undefined>;
  /** Last successful generation result per mode (audio + metadata) */
  lastResult: Record<string, Record<string, unknown> | undefined>;

  setPrompt: (mode: string, value: string) => void;
  setDraft: (mode: string, value: Record<string, unknown>) => void;
  clearDraft: (mode: string) => void;
  setHydration: (
    mode: string,
    data: {
      payload?: Record<string, unknown>;
      warnings?: unknown;
      paste_blocks?: unknown;
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
      drafts: {},
      lastResult: {},
      setPrompt: (mode, value) =>
        set((s) => ({ prompts: { ...s.prompts, [mode]: value } })),
      setDraft: (mode, value) =>
        set((s) => ({ drafts: { ...s.drafts, [mode]: value } })),
      clearDraft: (mode) =>
        set((s) => ({ drafts: { ...s.drafts, [mode]: undefined } })),
      setHydration: (mode, data) =>
        set((s) => ({
          payloads: { ...s.payloads, [mode]: data.payload },
          warnings: { ...s.warnings, [mode]: normalizeWarnings(data.warnings) },
          pasteBlocks: { ...s.pasteBlocks, [mode]: normalizePasteBlocks(data.paste_blocks) },
        })),
      setResult: (mode, result) =>
        set((s) => ({ lastResult: { ...s.lastResult, [mode]: result } })),
      reset: (mode) =>
        set((s) => ({
          prompts: { ...s.prompts, [mode]: "" },
          payloads: { ...s.payloads, [mode]: undefined },
          warnings: { ...s.warnings, [mode]: undefined },
          pasteBlocks: { ...s.pasteBlocks, [mode]: undefined },
          drafts: { ...s.drafts, [mode]: undefined },
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
        drafts: state.drafts,
      }),
    },
  ),
);
