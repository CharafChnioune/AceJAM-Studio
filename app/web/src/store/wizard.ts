import { create } from "zustand";
import { persist } from "zustand/middleware";
import { DEFAULT_AUDIO_BACKEND, normalizeAudioBackend } from "@/lib/audioBackend";

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
  setPasteBlocks: (mode: string, blocks: PasteBlock[] | undefined) => void;
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

function migrateAudioBackendDefaults(record: unknown): unknown {
  if (!record || typeof record !== "object" || Array.isArray(record)) return record;
  const next = { ...(record as Record<string, unknown>) };
  const hasAceStepModel = String(next.song_model ?? "").startsWith("acestep-v15-");
  if (hasAceStepModel || "audio_backend" in next || "use_mlx_dit" in next) {
    const backend = normalizeAudioBackend(
      next.audio_backend ?? (next.use_mlx_dit === true ? "mlx" : DEFAULT_AUDIO_BACKEND),
    );
    next.audio_backend = backend;
    next.use_mlx_dit = backend === "mlx";
  }
  return next;
}

function migratePersistedAudioDefaults(state: unknown): unknown {
  if (!state || typeof state !== "object") return state;
  const next = { ...(state as Record<string, unknown>) };
  for (const key of ["drafts", "payloads"]) {
    const bucket = next[key];
    if (!bucket || typeof bucket !== "object" || Array.isArray(bucket)) continue;
    next[key] = Object.fromEntries(
      Object.entries(bucket as Record<string, unknown>).map(([mode, value]) => [
        mode,
        migrateAudioBackendDefaults(value),
      ]),
    );
  }
  return next;
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
      setPasteBlocks: (mode, blocks) =>
        set((s) => ({
          pasteBlocks: {
            ...s.pasteBlocks,
            [mode]: blocks && blocks.length > 0 ? blocks : undefined,
          },
        })),
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
      version: 2,
      migrate: (state) => migratePersistedAudioDefaults(state) as WizardSlice,
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
