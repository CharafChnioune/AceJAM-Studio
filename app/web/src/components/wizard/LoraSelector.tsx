import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import { Layers } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
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
  normalizeLoraSelection,
  type LoraSelection,
} from "@/lib/lora";
import { cn } from "@/lib/utils";

const NONE = "__none__";

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
    onChange(loraSelectionFromAdapter(selected, scale));
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
          </div>
        )}
      </div>
      {selected && (
        <div className="space-y-2">
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
          {selection.lora_scale > 0.75 && (
            <p className="rounded-md border border-amber-400/30 bg-amber-400/10 p-2 text-xs text-amber-100">
              Volle LoRA-kracht kan vocals sneller vervormen. Voor testen is 0.45 meestal veiliger.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
