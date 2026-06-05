import type { LoraAdapter } from "@/lib/api";

export const MAX_MULTI_LORA_ADAPTERS = 4;
export const DEFAULT_LORA_SCALE = 1.0;

export interface SelectedAudioLoraAdapter {
  [key: string]: unknown;
  path: string;
  name: string;
  lora_adapter_path: string;
  lora_adapter_name: string;
  use_lora_trigger: boolean;
  lora_trigger_tag: string;
  lora_scale: number;
  adapter_model_variant: string;
  adapter_song_model: string;
}

export interface LoraSelection {
  use_lora: boolean;
  lora_adapter_path: string;
  lora_adapter_name: string;
  use_lora_trigger: boolean;
  lora_trigger_tag: string;
  lora_trigger_tags: string[];
  lora_scale: number;
  lora_adapters: SelectedAudioLoraAdapter[];
  adapter_model_variant: string;
  adapter_song_model: string;
}

export function emptyLoraSelection(): LoraSelection {
  return {
    use_lora: false,
    lora_adapter_path: "",
    lora_adapter_name: "",
    use_lora_trigger: false,
    lora_trigger_tag: "",
    lora_trigger_tags: [],
    lora_scale: DEFAULT_LORA_SCALE,
    lora_adapters: [],
    adapter_model_variant: "",
    adapter_song_model: "",
  };
}

export function songModelFromLoraVariant(variant: string | undefined): string {
  const normalized = String(variant || "").trim().toLowerCase().replace(/-/g, "_");
  if (normalized === "xl_sft") return "acestep-v15-xl-sft";
  if (normalized === "xl_base") return "acestep-v15-xl-base";
  if (normalized === "xl_turbo") return "acestep-v15-xl-turbo";
  if (normalized === "sft") return "acestep-v15-sft";
  if (normalized === "base") return "acestep-v15-base";
  if (normalized === "turbo") return "acestep-v15-turbo";
  return "";
}

export function loraAdapterLabel(adapter: Partial<LoraAdapter>): string {
  return String(
    adapter.display_name ||
      adapter.generation_trigger_tag ||
      adapter.trigger_tag ||
      adapter.label ||
      adapter.name ||
      "LoRA",
  );
}

export function isGenerationLoraAdapter(adapter: LoraAdapter): boolean {
  const adapterType = String(adapter.adapter_type || "").toLowerCase();
  return (
    adapterType === "lora" &&
    (adapter.generation_loadable === true || adapter.is_loadable === true) &&
    Boolean(adapter.path)
  );
}

function metadataText(adapter: Partial<LoraAdapter>, key: string): string {
  const value = adapter.metadata?.[key];
  return typeof value === "string" ? value.trim() : "";
}

function metadataList(adapter: Partial<LoraAdapter>, key: string): string[] {
  const value = adapter.metadata?.[key];
  return Array.isArray(value) ? value.map((item) => String(item || "").trim()).filter(Boolean) : [];
}

export function loraTriggerOptions(adapter: Partial<LoraAdapter>): string[] {
  const raw = [
    adapter.generation_trigger_tag,
    metadataText(adapter, "generation_trigger_tag"),
    adapter.trigger_tag,
    metadataText(adapter, "trigger_tag"),
    adapter.trigger_tag_raw,
    metadataText(adapter, "trigger_tag_raw"),
    adapter.trigger_aliases,
    metadataList(adapter, "trigger_aliases"),
    adapter.trigger_candidates,
    metadataList(adapter, "trigger_candidates"),
  ];
  const seen = new Set<string>();
  const options: string[] = [];
  for (const value of raw) {
    for (const part of String(value || "").split(/[,;\n]+/)) {
      const tag = part.replace(/\s+/g, " ").trim();
      if (!tag) continue;
      const key = tag.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      options.push(tag);
    }
  }
  return options;
}

export function loraSelectionFromAdapter(
  adapter: LoraAdapter,
  scale = DEFAULT_LORA_SCALE,
): LoraSelection {
  const entry = loraAdapterEntryFromAdapter(adapter, scale);
  return normalizeLoraSelection({
    use_lora: true,
    lora_adapter_path: entry.lora_adapter_path,
    lora_adapter_name: entry.lora_adapter_name,
    use_lora_trigger: entry.use_lora_trigger,
    lora_trigger_tag: entry.lora_trigger_tag,
    lora_scale: Number.isFinite(scale) ? scale : DEFAULT_LORA_SCALE,
    adapter_model_variant: entry.adapter_model_variant,
    adapter_song_model: entry.adapter_song_model,
    lora_adapters: [entry],
  });
}

export function loraAdapterEntryFromAdapter(
  adapter: LoraAdapter,
  scale = DEFAULT_LORA_SCALE,
): SelectedAudioLoraAdapter {
  const trigger = loraTriggerOptions(adapter)[0] || "";
  return {
    path: adapter.path,
    lora_adapter_path: adapter.path,
    name: loraAdapterLabel(adapter),
    lora_adapter_name: loraAdapterLabel(adapter),
    use_lora_trigger: Boolean(trigger),
    lora_trigger_tag: trigger,
    lora_scale: Number.isFinite(scale) ? scale : DEFAULT_LORA_SCALE,
    adapter_model_variant: adapter.model_variant || "",
    adapter_song_model: adapter.song_model || songModelFromLoraVariant(adapter.model_variant),
  };
}

function normalizeAdapterEntry(
  entry: Partial<SelectedAudioLoraAdapter> | undefined,
  fallbackScale: number,
): SelectedAudioLoraAdapter | null {
  const path = String(entry?.path || entry?.lora_adapter_path || "").trim();
  if (!path) return null;
  const trigger = String(entry?.lora_trigger_tag || "").trim();
  const scale = Number(entry?.lora_scale ?? fallbackScale);
  return {
    path,
    lora_adapter_path: path,
    name: String(entry?.name || entry?.lora_adapter_name || "").trim(),
    lora_adapter_name: String(entry?.lora_adapter_name || entry?.name || "").trim(),
    use_lora_trigger: Boolean(trigger && entry?.use_lora_trigger !== false),
    lora_trigger_tag: trigger,
    lora_scale: Number.isFinite(scale) ? scale : fallbackScale,
    adapter_model_variant: String(entry?.adapter_model_variant || "").trim(),
    adapter_song_model: String(entry?.adapter_song_model || "").trim(),
  };
}

export function normalizeLoraSelection(
  selection: Partial<LoraSelection> | undefined,
): LoraSelection {
  const rawScale = Number(selection?.lora_scale ?? DEFAULT_LORA_SCALE);
  const scale = Number.isFinite(rawScale) ? rawScale : DEFAULT_LORA_SCALE;
  const adapterEntries = Array.isArray(selection?.lora_adapters)
    ? selection.lora_adapters
        .map((entry) => normalizeAdapterEntry(entry, scale))
        .filter((entry): entry is SelectedAudioLoraAdapter => Boolean(entry))
        .slice(0, MAX_MULTI_LORA_ADAPTERS)
    : [];
  const legacyPath = String(selection?.lora_adapter_path || "").trim();
  if (legacyPath && !adapterEntries.some((entry) => entry.path === legacyPath)) {
    const legacy = normalizeAdapterEntry(
      {
        path: legacyPath,
        lora_adapter_name: selection?.lora_adapter_name,
        use_lora_trigger: selection?.use_lora_trigger,
        lora_trigger_tag: selection?.lora_trigger_tag,
        lora_scale: scale,
        adapter_model_variant: selection?.adapter_model_variant,
        adapter_song_model: selection?.adapter_song_model,
      },
      scale,
    );
    if (legacy) adapterEntries.unshift(legacy);
  }
  const first = adapterEntries[0];
  const use = Boolean(selection?.use_lora && first);
  const trigger = String(selection?.lora_trigger_tag || "").trim();
  const triggerTags = adapterEntries
    .map((entry) => String(entry.lora_trigger_tag || "").trim())
    .filter(Boolean)
    .filter((tag, index, all) => all.findIndex((item) => item.toLowerCase() === tag.toLowerCase()) === index);
  const primaryTrigger = trigger || first?.lora_trigger_tag || "";
  const useTrigger = Boolean(use && triggerTags.length > 0 && selection?.use_lora_trigger !== false);
  return {
    use_lora: use,
    lora_adapter_path: use ? first.path : "",
    lora_adapter_name: use
      ? adapterEntries.length > 1
        ? `${adapterEntries.length} LoRAs`
        : String(first.lora_adapter_name || first.name || "").trim()
      : "",
    use_lora_trigger: useTrigger,
    lora_trigger_tag: use ? primaryTrigger : "",
    lora_trigger_tags: use ? triggerTags : [],
    lora_scale: use ? scale : DEFAULT_LORA_SCALE,
    lora_adapters: use ? adapterEntries : [],
    adapter_model_variant: use
      ? String(first.adapter_model_variant || selection?.adapter_model_variant || "").trim()
      : "",
    adapter_song_model: use
      ? String(first.adapter_song_model || selection?.adapter_song_model || "").trim()
      : "",
  };
}
