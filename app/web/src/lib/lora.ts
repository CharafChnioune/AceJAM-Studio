import type { LoraAdapter } from "@/lib/api";

export interface LoraSelection {
  use_lora: boolean;
  lora_adapter_path: string;
  lora_adapter_name: string;
  lora_scale: number;
  adapter_model_variant: string;
}

export const DEFAULT_LORA_SCALE = 0.45;

export function emptyLoraSelection(): LoraSelection {
  return {
    use_lora: false,
    lora_adapter_path: "",
    lora_adapter_name: "",
    lora_scale: DEFAULT_LORA_SCALE,
    adapter_model_variant: "",
  };
}

export function loraAdapterLabel(adapter: Partial<LoraAdapter>): string {
  return String(
    adapter.display_name ||
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

export function loraSelectionFromAdapter(
  adapter: LoraAdapter,
  scale = DEFAULT_LORA_SCALE,
): LoraSelection {
  return {
    use_lora: true,
    lora_adapter_path: adapter.path,
    lora_adapter_name: loraAdapterLabel(adapter),
    lora_scale: Number.isFinite(scale) ? scale : DEFAULT_LORA_SCALE,
    adapter_model_variant: adapter.model_variant || "",
  };
}

export function normalizeLoraSelection(
  selection: Partial<LoraSelection> | undefined,
): LoraSelection {
  const path = String(selection?.lora_adapter_path || "").trim();
  const use = Boolean(selection?.use_lora && path);
  const rawScale = Number(selection?.lora_scale ?? DEFAULT_LORA_SCALE);
  return {
    use_lora: use,
    lora_adapter_path: use ? path : "",
    lora_adapter_name: use ? String(selection?.lora_adapter_name || "").trim() : "",
    lora_scale: use && Number.isFinite(rawScale) ? rawScale : DEFAULT_LORA_SCALE,
    adapter_model_variant: use
      ? String(selection?.adapter_model_variant || "").trim()
      : "",
  };
}
