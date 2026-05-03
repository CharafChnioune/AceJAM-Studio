import * as React from "react";
import { motion } from "framer-motion";
import { Zap, CircleCheck, Crown } from "lucide-react";
import { cn } from "@/lib/utils";

interface PresetDef {
  key: "draft" | "standard" | "chart_master";
  label: string;
  blurb: string;
  steps: number;
  icon: React.ComponentType<{ className?: string }>;
}

const PRESETS: PresetDef[] = [
  { key: "draft", label: "Draft", blurb: "Snel, ~8 stappen", steps: 8, icon: Zap },
  { key: "standard", label: "Standard", blurb: "Gebalanceerd, 32 stappen", steps: 32, icon: CircleCheck },
  { key: "chart_master", label: "Chart Master", blurb: "Maximaal, 64 stappen", steps: 64, icon: Crown },
];

interface QualityPresetsProps {
  value: "draft" | "standard" | "chart_master";
  onChange: (
    profile: "draft" | "standard" | "chart_master",
    inferenceSteps: number,
  ) => void;
  className?: string;
}

export function QualityPresets({ value, onChange, className }: QualityPresetsProps) {
  return (
    <div className={cn("grid grid-cols-3 gap-2", className)}>
      {PRESETS.map((p) => {
        const active = p.key === value;
        const Icon = p.icon;
        return (
          <motion.button
            key={p.key}
            type="button"
            onClick={() => onChange(p.key, p.steps)}
            whileTap={{ scale: 0.97 }}
            className={cn(
              "group relative overflow-hidden rounded-xl border p-3 text-left transition-colors",
              active
                ? "border-primary/60 bg-primary/15 text-primary-foreground"
                : "border-border bg-card/40 text-foreground hover:border-foreground/30",
            )}
          >
            <div className="flex items-center justify-between">
              <Icon className={cn("size-4", active && "text-primary")} />
              {active && (
                <motion.div
                  layoutId="quality-preset-pill"
                  className="absolute inset-0 -z-10 rounded-xl bg-primary/10"
                />
              )}
            </div>
            <p className="mt-2 font-display text-sm font-semibold">{p.label}</p>
            <p className="text-[10px] text-muted-foreground">{p.blurb}</p>
          </motion.button>
        );
      })}
    </div>
  );
}
