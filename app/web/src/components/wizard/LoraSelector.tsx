import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Layers, Tag } from "lucide-react";

import { Badge } from "@/components/ui/badge";
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
import { getLoraAdapters } from "@/lib/api";
import {
  emptyLoraSelection,
  isGenerationLoraAdapter,
  loraAdapterLabel,
  loraSelectionFromAdapter,
  loraTriggerOptions,
  normalizeLoraSelection,
  type LoraSelection,
} from "@/lib/lora";
import { cn } from "@/lib/utils";

const NONE = "__none__";
const NO_TRIGGER = "__no_trigger__";

interface LoraSelectorProps {
  value: Partial<LoraSelection>;
  onChange: (value: LoraSelection) => void;
  className?: string;
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
  const selected = adapters.find((adapter) => adapter.path === selection.lora_adapter_path);
  const selectedValue = selected ? selected.path : NONE;
  const triggerOptions = React.useMemo(
    () => (selected ? loraTriggerOptions(selected) : []),
    [selected],
  );
  const effectiveTrigger = selection.lora_trigger_tag || triggerOptions[0] || "";
  const effectiveUseTrigger = Boolean(
    selected && effectiveTrigger && (selection.use_lora_trigger || !selection.lora_trigger_tag),
  );

  React.useEffect(() => {
    if (!selected || !selection.use_lora || selection.lora_trigger_tag || triggerOptions.length === 0) {
      return;
    }
    onChange({
      ...loraSelectionFromAdapter(selected, selection.lora_scale),
      use_lora_trigger: true,
      lora_trigger_tag: triggerOptions[0],
    });
  }, [
    selected,
    selection.use_lora,
    selection.lora_trigger_tag,
    selection.lora_scale,
    triggerOptions,
    onChange,
  ]);

  const selectionForSelected = (patch: Partial<LoraSelection> = {}): LoraSelection | null => {
    if (!selected) return null;
    return {
      ...loraSelectionFromAdapter(selected, selection.lora_scale),
      use_lora_trigger: effectiveUseTrigger,
      lora_trigger_tag: effectiveTrigger,
      ...patch,
    };
  };

  const setAdapter = (path: string) => {
    if (path === NONE) {
      onChange(emptyLoraSelection());
      return;
    }
    const adapter = adapters.find((item) => item.path === path);
    if (adapter) onChange(loraSelectionFromAdapter(adapter, selection.lora_scale));
  };

  const setScale = (scale: number) => {
    if (!selected) return;
    const next = selectionForSelected({ lora_scale: scale });
    if (next) onChange(next);
  };

  const setUseTrigger = (enabled: boolean) => {
    const next = selectionForSelected({
      use_lora_trigger: enabled && Boolean(effectiveTrigger),
      lora_trigger_tag: effectiveTrigger,
    });
    if (next) onChange(next);
  };

  const setTriggerTag = (tag: string) => {
    const trigger = tag === NO_TRIGGER ? "" : tag;
    const next = selectionForSelected({
      use_lora_trigger: Boolean(trigger),
      lora_trigger_tag: trigger,
    });
    if (next) onChange(next);
  };

  return (
    <div className={cn("space-y-3", className)}>
      <div className="space-y-1.5">
        <Label className="flex items-center gap-2">
          <Layers className="size-4 text-primary" />
          LoRA adapter
        </Label>
        <Select
          value={selectedValue}
          onValueChange={setAdapter}
          disabled={adaptersQuery.isLoading}
        >
          <SelectTrigger>
            <SelectValue
              placeholder={adaptersQuery.isLoading ? "LoRA's laden..." : "Geen LoRA"}
            />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={NONE}>Geen LoRA</SelectItem>
            {adapters.map((adapter) => (
              <SelectItem key={adapter.path} value={adapter.path}>
                <div className="flex min-w-0 items-center gap-2">
                  <span className="truncate">{loraAdapterLabel(adapter)}</span>
                  {(adapter.model_variant || adapter.song_model) && (
                    <span className="text-[10px] text-muted-foreground">
                      {adapter.model_variant || adapter.song_model}
                    </span>
                  )}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {!adaptersQuery.isLoading && adapters.length === 0 && (
          <p className="text-xs text-muted-foreground">
            Geen generation-loadable PEFT LoRA adapters gevonden.
          </p>
        )}
        {selected && (
          <div className="flex flex-wrap gap-1.5">
            <Badge variant="secondary">{selected.adapter_type || "lora"}</Badge>
            {(selected.model_variant || selected.song_model) && (
              <Badge variant="outline">
                {selected.model_variant || selected.song_model}
              </Badge>
            )}
            {selected.trigger_tag && (
              <Badge variant="outline">{selected.trigger_tag}</Badge>
            )}
            {selected.trigger_source && (
              <Badge variant={selected.trigger_source === "missing" ? "destructive" : "secondary"}>
                trigger: {selected.trigger_source}
              </Badge>
            )}
          </div>
        )}
      </div>
      {selected && (
        <div className="space-y-4">
          <div className="space-y-2 rounded-md border border-border/70 p-3">
            <div className="flex items-center justify-between gap-3">
              <Label className="flex items-center gap-2">
                <Tag className="size-4 text-primary" />
                Trigger tag activeren
              </Label>
              <Switch
                checked={effectiveUseTrigger}
                disabled={triggerOptions.length === 0}
                onCheckedChange={setUseTrigger}
              />
            </div>
            {triggerOptions.length > 1 ? (
              <Select
                value={effectiveTrigger || NO_TRIGGER}
                onValueChange={setTriggerTag}
                disabled={!effectiveUseTrigger}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Kies trigger tag" />
                </SelectTrigger>
                <SelectContent>
                  {triggerOptions.map((tag) => (
                    <SelectItem key={tag} value={tag}>
                      {tag}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : triggerOptions.length === 1 ? (
              <div className="rounded-md border bg-background/40 px-3 py-2 text-sm font-mono">
                {triggerOptions[0]}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">
                Deze adapter heeft geen trigger-tag metadata.
              </p>
            )}
            <p className="text-xs text-muted-foreground">
              De tag wordt aan de ACE-Step caption toegevoegd, niet aan de lyrics.
            </p>
          </div>
          <div className="flex items-baseline justify-between">
            <Label>LoRA scale (jouw keuze)</Label>
            <span className="font-mono text-xs">{selection.lora_scale.toFixed(2)}</span>
          </div>
          <Slider
            value={[selection.lora_scale]}
            min={0}
            max={1}
            step={0.01}
            onValueChange={(next) => setScale(next[0] ?? 1)}
          />
          {selection.lora_scale > 0.75 && (
            <p className="rounded-md border border-amber-400/30 bg-amber-400/10 p-2 text-xs text-amber-100">
              Volle LoRA-kracht gebruikt exact deze scale in de render. Als vocals vervormen, vergelijk dezelfde seed met een lagere scale.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
