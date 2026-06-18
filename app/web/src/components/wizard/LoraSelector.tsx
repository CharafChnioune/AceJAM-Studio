import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, Layers, Plus, Tag, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getLoraAdapters, type LoraAdapter } from "@/lib/api";
import {
  emptyLoraSelection,
  isGenerationLoraAdapter,
  loraAdapterEntryFromAdapter,
  loraAdapterLabel,
  loraSelectionFromAdapter,
  loraTriggerOptions,
  MAX_MULTI_LORA_ADAPTERS,
  normalizeLoraSelection,
  type LoraSelection,
  type SelectedAudioLoraAdapter,
} from "@/lib/lora";
import { cn } from "@/lib/utils";

const NO_TRIGGER = "__no_trigger__";

interface LoraSelectorProps {
  value: Partial<LoraSelection>;
  onChange: (value: LoraSelection) => void;
  className?: string;
}

function entryFromSelectedAdapter(adapter: LoraAdapter, scale: number): SelectedAudioLoraAdapter {
  return loraAdapterEntryFromAdapter(adapter, scale);
}

export function LoraSelector({ value, onChange, className }: LoraSelectorProps) {
  const selection = normalizeLoraSelection(value);
  const adaptersQuery = useQuery({
    queryKey: ["lora", "adapters"],
    queryFn: getLoraAdapters,
    staleTime: 30_000,
  });

  const adapters = React.useMemo(
    () => (adaptersQuery.data?.adapters ?? []).filter(isGenerationLoraAdapter),
    [adaptersQuery.data?.adapters],
  );
  const adaptersError = React.useMemo(() => {
    if (!adaptersQuery.error) return "";
    if (adaptersQuery.error instanceof Error) return adaptersQuery.error.message;
    return "LoRA adapters laden mislukte.";
  }, [adaptersQuery.error]);
  const selectedPaths = React.useMemo(
    () => new Set(selection.lora_adapters.map((adapter) => adapter.path)),
    [selection.lora_adapters],
  );
  const selectedAdapters = React.useMemo(
    () => selection.lora_adapters
      .map((entry) => adapters.find((adapter) => adapter.path === entry.path))
      .filter((adapter): adapter is LoraAdapter => Boolean(adapter)),
    [adapters, selection.lora_adapters],
  );
  const primary = selectedAdapters[0];
  const primaryTriggerOptions = React.useMemo(
    () => (primary ? loraTriggerOptions(primary) : []),
    [primary],
  );
  const selectedCount = selection.lora_adapters.length;
  const effectiveTrigger = selection.lora_trigger_tag || primaryTriggerOptions[0] || "";
  const effectiveUseTrigger = Boolean(selection.use_lora && selection.lora_trigger_tags.length > 0 && selection.use_lora_trigger !== false);

  React.useEffect(() => {
    if (!primary || !selection.use_lora || selection.lora_trigger_tags.length > 0) return;
    const next = loraSelectionFromAdapter(primary, selection.lora_scale);
    onChange(normalizeLoraSelection({
      ...next,
      lora_adapters: selection.lora_adapters.map((entry, index) => ({
        ...entry,
        ...(index === 0
          ? {
              use_lora_trigger: true,
              lora_trigger_tag: primaryTriggerOptions[0] || entry.lora_trigger_tag,
            }
          : {}),
      })),
    }));
  }, [
    primary,
    primaryTriggerOptions,
    selection.lora_adapters,
    selection.lora_scale,
    selection.lora_trigger_tags.length,
    selection.use_lora,
    onChange,
  ]);

  const emit = (entries: SelectedAudioLoraAdapter[], patch: Partial<LoraSelection> = {}) => {
    const nextScale = Number(patch.lora_scale ?? selection.lora_scale);
    const next = normalizeLoraSelection({
      ...selection,
      ...patch,
      use_lora: entries.length > 0,
      lora_adapters: entries,
      lora_adapter_path: entries[0]?.path || "",
      lora_adapter_name: entries.length > 1 ? `${entries.length} LoRAs` : entries[0]?.lora_adapter_name || "",
      adapter_model_variant: entries[0]?.adapter_model_variant || "",
      adapter_song_model: entries[0]?.adapter_song_model || "",
      lora_trigger_tag: entries[0]?.lora_trigger_tag || "",
      lora_scale: Number.isFinite(nextScale) ? nextScale : selection.lora_scale,
    });
    onChange(next);
  };

  const toggleAdapter = (adapter: LoraAdapter) => {
    if (selectedPaths.has(adapter.path)) {
      emit(selection.lora_adapters.filter((entry) => entry.path !== adapter.path));
      return;
    }
    if (selection.lora_adapters.length >= MAX_MULTI_LORA_ADAPTERS) return;
    emit([...selection.lora_adapters, entryFromSelectedAdapter(adapter, selection.lora_scale)]);
  };

  const removeAdapter = (path: string) => {
    emit(selection.lora_adapters.filter((entry) => entry.path !== path));
  };

  const clear = () => onChange(emptyLoraSelection());

  const setScale = (scale: number) => {
    const entries = selection.lora_adapters.map((entry) => ({ ...entry, lora_scale: scale }));
    emit(entries, { lora_scale: scale });
  };

  const setUseTrigger = (enabled: boolean) => {
    const entries = selection.lora_adapters.map((entry) => ({
      ...entry,
      use_lora_trigger: enabled && Boolean(entry.lora_trigger_tag),
    }));
    emit(entries, { use_lora_trigger: enabled });
  };

  const setTriggerTag = (tag: string) => {
    if (!primary) return;
    const trigger = tag === NO_TRIGGER ? "" : tag;
    const entries = selection.lora_adapters.map((entry, index) => index === 0
      ? {
          ...entry,
          use_lora_trigger: Boolean(trigger),
          lora_trigger_tag: trigger,
        }
      : entry);
    emit(entries, {
      use_lora_trigger: Boolean(trigger),
      lora_trigger_tag: trigger,
    });
  };

  return (
    <div className={cn("space-y-3", className)}>
      <div className="space-y-1.5">
        <div className="flex items-center justify-between gap-3">
          <Label className="flex items-center gap-2">
            <Layers className="size-4 text-primary" />
            LoRA adapters
          </Label>
          <div className="flex items-center gap-2">
            <Badge variant={selectedCount > 0 ? "secondary" : "outline"}>
              {selectedCount}/{MAX_MULTI_LORA_ADAPTERS}
            </Badge>
            {selectedCount > 0 && (
              <Button type="button" variant="ghost" size="sm" onClick={clear}>
                <X className="size-4" />
              </Button>
            )}
          </div>
        </div>

        <div className="grid max-h-64 gap-2 overflow-auto rounded-md border border-border/70 p-2 sm:grid-cols-2">
          {adaptersQuery.isLoading ? (
            <p className="px-2 py-3 text-sm text-muted-foreground">LoRA's laden...</p>
          ) : adaptersError ? (
            <div className="space-y-2 px-2 py-3 text-sm">
              <p className="text-amber-500">
                LoRA adapters konden niet worden geladen.
              </p>
              <p className="text-xs text-muted-foreground">
                {adaptersError}
              </p>
            </div>
          ) : adapters.length === 0 ? (
            <p className="px-2 py-3 text-sm text-muted-foreground">
              Geen generation-loadable PEFT LoRA adapters gevonden.
            </p>
          ) : (
            adapters.map((adapter) => {
              const active = selectedPaths.has(adapter.path);
              const disabled = !active && selectedCount >= MAX_MULTI_LORA_ADAPTERS;
              return (
                <Button
                  key={adapter.path}
                  type="button"
                  variant={active ? "secondary" : "outline"}
                  className="h-auto justify-start gap-2 px-2 py-2 text-left"
                  disabled={disabled}
                  onClick={() => toggleAdapter(adapter)}
                >
                  {active ? <Check className="size-4 shrink-0 text-primary" /> : <Plus className="size-4 shrink-0" />}
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm">{loraAdapterLabel(adapter)}</span>
                    {(adapter.model_variant || adapter.song_model || adapter.trigger_tag) && (
                      <span className="block truncate text-[11px] text-muted-foreground">
                        {[adapter.model_variant || adapter.song_model, adapter.trigger_tag].filter(Boolean).join(" · ")}
                      </span>
                    )}
                  </span>
                </Button>
              );
            })
          )}
        </div>

        {selectedCount > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {selection.lora_adapters.map((entry) => (
              <Badge key={entry.path} variant="secondary" className="gap-1">
                {entry.lora_adapter_name || entry.name || "LoRA"}
                <button type="button" onClick={() => removeAdapter(entry.path)} aria-label="LoRA verwijderen">
                  <X className="size-3" />
                </button>
              </Badge>
            ))}
          </div>
        )}
      </div>

      {selectedCount > 0 && (
        <div className="space-y-4">
          <div className="space-y-2 rounded-md border border-border/70 p-3">
            <div className="flex items-center justify-between gap-3">
              <Label className="flex items-center gap-2">
                <Tag className="size-4 text-primary" />
                Trigger tags activeren
              </Label>
              <Switch
                checked={effectiveUseTrigger}
                disabled={selection.lora_trigger_tags.length === 0}
                onCheckedChange={setUseTrigger}
              />
            </div>
            {selectedCount === 1 && primaryTriggerOptions.length > 1 ? (
              <Select
                value={effectiveTrigger || NO_TRIGGER}
                onValueChange={setTriggerTag}
                disabled={!effectiveUseTrigger}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Kies trigger tag" />
                </SelectTrigger>
                <SelectContent>
                  {primaryTriggerOptions.map((tag) => (
                    <SelectItem key={tag} value={tag}>
                      {tag}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : selection.lora_trigger_tags.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {selection.lora_trigger_tags.map((tag) => (
                  <Badge key={tag} variant="outline">{tag}</Badge>
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">
                Deze adapterselectie heeft geen trigger-tag metadata.
              </p>
            )}
            <p className="text-xs text-muted-foreground">
              Tags worden aan de ACE-Step caption toegevoegd, niet aan de lyrics.
            </p>
          </div>

          <div className="flex items-baseline justify-between">
            <Label>LoRA scale</Label>
            <span className="font-mono text-xs">{selection.lora_scale.toFixed(2)}</span>
          </div>
          <Slider
            value={[selection.lora_scale]}
            min={0}
            max={1}
            step={0.01}
            onValueChange={(next) => setScale(next[0] ?? 1)}
          />
          {selectedCount > 1 && (
            <p className="rounded-md border border-primary/20 bg-primary/10 p-2 text-xs text-primary">
              Meerdere audio-LoRAs worden als één MLX adapter gemerged. Getest maximum: {MAX_MULTI_LORA_ADAPTERS}.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
