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
          return { label: `Block ${index + 1}`, content: item };
        }
        if (item && typeof item === "object") {
          const record = item as Record<string, unknown>;
          const content = String(record.content ?? record.text ?? "");
          const label = String(
            record.label ?? record.title ?? `Block ${index + 1}`,
          ).trim();
          return { label, content };
        }
        return undefined;
      })
      .filter((item): item is PasteBlock => Boolean(item));
  }
  if (typeof value === "string") {
    return [{ label: "Paste block", content: value }];
  }
  return [];
}

interface WizardSlice {
  /** Per wizard mode: prompt + last AI-fill response + warnings */
  prompts: Record<string, string>;
  promptPresets: Record<string, string | undefined>;
  payloads: Record<string, Record<string, unknown> | undefined>;
  validations: Record<string, Record<string, unknown> | undefined>;
  warnings: Record<string, string[] | undefined>;
  pasteBlocks: Record<string, PasteBlock[] | undefined>;
  companions: Record<string, Record<string, unknown> | undefined>;
  queues: Record<string, Record<string, unknown>[] | undefined>;
  /** Persisted form values per wizard mode. */
  drafts: Record<string, Record<string, unknown> | undefined>;
  /** Last successful generation result per mode (audio + metadata) */
  lastResult: Record<string, Record<string, unknown> | undefined>;

  setPrompt: (mode: string, value: string) => void;
  setPromptPreset: (mode: string, value: string | undefined) => void;
  setDraft: (mode: string, value: Record<string, unknown>) => void;
  clearDraft: (mode: string) => void;
  setPasteBlocks: (mode: string, blocks: PasteBlock[] | undefined) => void;
  setHydration: (
    mode: string,
    data: {
      payload?: Record<string, unknown>;
      warnings?: unknown;
      validation?: Record<string, unknown> | null;
      paste_blocks?: unknown;
      companion?: Record<string, unknown> | null;
      queue?: Record<string, unknown>[] | null;
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

function clearLegacyBrokenLoraSweepQueue(state: unknown): unknown {
  if (!state || typeof state !== "object") return state;
  const next = { ...(state as Record<string, unknown>) };
  for (const key of ["queues", "payloads", "validations", "warnings", "pasteBlocks", "companions"]) {
    const bucket = next[key];
    if (!bucket || typeof bucket !== "object" || Array.isArray(bucket)) continue;
    next[key] = { ...(bucket as Record<string, unknown>), lora_sweep: undefined };
  }
  return next;
}

export const useWizardStore = create<WizardSlice>()(
  persist(
    (set) => ({
      prompts: {},
      promptPresets: {},
      payloads: {},
      validations: {},
      warnings: {},
      pasteBlocks: {},
      companions: {},
      queues: {},
      drafts: {},
      lastResult: {},
      setPrompt: (mode, value) =>
        set((s) => ({ prompts: { ...s.prompts, [mode]: value } })),
      setPromptPreset: (mode, value) =>
        set((s) => ({
          promptPresets: {
            ...s.promptPresets,
            [mode]: value && value.trim() ? value.trim() : undefined,
          },
        })),
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
          validations: { ...s.validations, [mode]: data.validation ?? undefined },
          warnings: { ...s.warnings, [mode]: normalizeWarnings(data.warnings) },
          pasteBlocks: { ...s.pasteBlocks, [mode]: normalizePasteBlocks(data.paste_blocks) },
          companions: {
            ...s.companions,
            [mode]: data.companion && Object.keys(data.companion).length > 0 ? data.companion : undefined,
          },
          queues: {
            ...s.queues,
            [mode]: data.queue && data.queue.length > 0 ? data.queue : undefined,
          },
        })),
      setResult: (mode, result) =>
        set((s) => ({ lastResult: { ...s.lastResult, [mode]: result } })),
      reset: (mode) =>
        set((s) => ({
          prompts: { ...s.prompts, [mode]: "" },
          promptPresets: { ...s.promptPresets, [mode]: undefined },
          payloads: { ...s.payloads, [mode]: undefined },
          validations: { ...s.validations, [mode]: undefined },
          warnings: { ...s.warnings, [mode]: undefined },
          pasteBlocks: { ...s.pasteBlocks, [mode]: undefined },
          companions: { ...s.companions, [mode]: undefined },
          queues: { ...s.queues, [mode]: undefined },
          drafts: { ...s.drafts, [mode]: undefined },
          lastResult: { ...s.lastResult, [mode]: undefined },
        })),
    }),
    {
      name: "acejam-wizard-state",
      version: 6,
      migrate: (state) => clearLegacyBrokenLoraSweepQueue(migratePersistedAudioDefaults(state)) as WizardSlice,
      partialize: (state) => ({
        prompts: state.prompts,
        promptPresets: state.promptPresets,
        payloads: state.payloads,
        validations: state.validations,
        warnings: state.warnings,
        pasteBlocks: state.pasteBlocks,
        companions: state.companions,
        queues: state.queues,
        drafts: state.drafts,
      }),
    },
  ),
);
