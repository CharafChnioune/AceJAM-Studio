import * as React from "react";
import { ChevronDown, Copy, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ReviewStepProps {
  payload: Record<string, unknown>;
  warnings?: string[];
  primaryFields: Array<{ key: string; label: string; format?: (v: unknown) => string }>;
  className?: string;
}

export function ReviewStep({ payload, warnings = [], primaryFields, className }: ReviewStepProps) {
  const [showJson, setShowJson] = React.useState(false);
  const [copied, setCopied] = React.useState(false);

  const copyJson = async () => {
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className={cn("space-y-4", className)}>
      <div className="grid gap-3 sm:grid-cols-2">
        {primaryFields.map(({ key, label, format }) => {
          const v = payload[key];
          const display = format ? format(v) : v == null ? "—" : String(v);
          return (
            <div
              key={key}
              className="rounded-lg border border-border/60 bg-card/40 p-3"
            >
              <p className="text-[10px] uppercase tracking-widest text-muted-foreground">
                {label}
              </p>
              <p className="mt-1 truncate text-sm font-medium" title={display}>
                {display || "—"}
              </p>
            </div>
          );
        })}
      </div>

      {warnings.length > 0 && (
        <div className="rounded-xl border border-yellow-500/30 bg-yellow-500/10 p-4 text-sm text-yellow-100">
          <p className="mb-1 text-xs font-semibold uppercase tracking-widest text-yellow-300">
            Aandachtspunten
          </p>
          <ul className="list-disc space-y-0.5 pl-5">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <details
        open={showJson}
        onToggle={(e) => setShowJson((e.target as HTMLDetailsElement).open)}
        className="rounded-xl border bg-card/40"
      >
        <summary className="flex cursor-pointer items-center justify-between gap-3 px-4 py-3 text-sm font-medium">
          <span className="flex items-center gap-2">
            <ChevronDown
              className={cn(
                "size-4 transition-transform",
                showJson && "rotate-180",
              )}
            />
            Volledige payload
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.preventDefault();
              copyJson();
            }}
            className="gap-1.5 text-xs"
          >
            {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
            {copied ? "Gekopieerd" : "Kopieer"}
          </Button>
        </summary>
        <pre className="max-h-[420px] overflow-auto px-4 pb-4 text-[11px] leading-relaxed text-muted-foreground">
          {JSON.stringify(payload, null, 2)}
        </pre>
      </details>
    </div>
  );
}
