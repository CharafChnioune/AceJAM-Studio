import type { LoraAdapter } from "@/lib/api";

export interface LoraSelection {
  use_lora: boolean;
  lora_adapter_path: string;
  lora_adapter_name: string;
  use_lora_trigger: boolean;
  lora_trigger_tag: string;
  lora_scale: number;
  adapter_model_variant: string;
  adapter_song_model: string;
}

export const DEFAULT_LORA_SCALE = 0.45;

export function emptyLoraSelection(): LoraSelection {
  return {
    use_lora: false,
    lora_adapter_path: "",
    lora_adapter_name: "",
    use_lora_trigger: false,
    lora_trigger_tag: "",
    lora_scale: DEFAULT_LORA_SCALE,
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
  const trigger = loraTriggerOptions(adapter)[0] || "";
  return {
    use_lora: true,
    lora_adapter_path: adapter.path,
    lora_adapter_name: loraAdapterLabel(adapter),
    use_lora_trigger: Boolean(trigger),
    lora_trigger_tag: trigger,
    lora_scale: Number.isFinite(scale) ? scale : DEFAULT_LORA_SCALE,
    adapter_model_variant: adapter.model_variant || "",
    adapter_song_model: adapter.song_model || songModelFromLoraVariant(adapter.model_variant),
  };
}

export function normalizeLoraSelection(
  selection: Partial<LoraSelection> | undefined,
): LoraSelection {
  const path = String(selection?.lora_adapter_path || "").trim();
  const use = Boolean(selection?.use_lora && path);
  const rawScale = Number(selection?.lora_scale ?? DEFAULT_LORA_SCALE);
  const trigger = String(selection?.lora_trigger_tag || "").trim();
  const useTrigger = Boolean(use && trigger && selection?.use_lora_trigger !== false);
  return {
    use_lora: use,
    lora_adapter_path: use ? path : "",
    lora_adapter_name: use ? String(selection?.lora_adapter_name || "").trim() : "",
    use_lora_trigger: useTrigger,
    lora_trigger_tag: use ? trigger : "",
    lora_scale: use && Number.isFinite(rawScale) ? rawScale : DEFAULT_LORA_SCALE,
    adapter_model_variant: use
      ? String(selection?.adapter_model_variant || "").trim()
      : "",
    adapter_song_model: use
      ? String(selection?.adapter_song_model || "").trim()
      : "",
  };
}
