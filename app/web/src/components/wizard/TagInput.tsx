import * as React from "react";
import { X, Plus } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

interface TagInputProps {
  value: unknown;
  onChange: (value: string) => void;
  placeholder?: string;
  suggestions?: string[];
  className?: string;
  variant?: "default" | "negative";
}

function parse(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item ?? "").trim()).filter(Boolean);
  }
  if (value === null || value === undefined) return [];
  const raw = String(value);
  if (!raw) return [];
  return raw
    .split(/[,\n]/)
    .map((t) => t.trim())
    .filter(Boolean);
}

export function TagInput({
  value,
  onChange,
  placeholder = "Voeg een tag toe en druk op Enter",
  suggestions = [],
  className,
  variant = "default",
}: TagInputProps) {
  const [draft, setDraft] = React.useState("");
  const tags = parse(value);

  const commit = (raw: string) => {
    const next = parse(raw);
    if (!next.length) return;
    const merged = Array.from(new Set([...tags, ...next]));
    onChange(merged.join(", "));
    setDraft("");
  };

  const remove = (idx: number) => {
    const next = tags.filter((_, i) => i !== idx);
    onChange(next.join(", "));
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === "," || e.key === "Tab") {
      if (draft.trim()) {
        e.preventDefault();
        commit(draft);
      }
    } else if (e.key === "Backspace" && !draft && tags.length) {
      onChange(tags.slice(0, -1).join(", "));
    }
  };

  const tagClass =
    variant === "negative"
      ? "bg-destructive/15 text-destructive-foreground border-destructive/30"
      : "bg-primary/15 text-primary-foreground border-primary/30";

  const filteredSuggestions = suggestions
    .filter((s) => !tags.includes(s) && s.toLowerCase().includes(draft.toLowerCase()))
    .slice(0, 8);

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex min-h-9 flex-wrap items-center gap-1.5 rounded-md border bg-transparent px-2 py-1.5 transition-colors focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/30">
        <AnimatePresence initial={false}>
          {tags.map((t, idx) => (
            <motion.span
              key={`${t}-${idx}`}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={{ duration: 0.12 }}
              className={cn(
                "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium",
                tagClass,
              )}
            >
              {t}
              <button
                type="button"
                onClick={() => remove(idx)}
                className="rounded-full opacity-60 transition-opacity hover:opacity-100"
                aria-label={`Verwijder ${t}`}
              >
                <X className="size-3" />
              </button>
            </motion.span>
          ))}
        </AnimatePresence>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          onBlur={() => draft.trim() && commit(draft)}
          placeholder={tags.length === 0 ? placeholder : "+"}
          className="min-w-[60px] flex-1 bg-transparent py-0.5 text-sm placeholder:text-muted-foreground focus:outline-none"
        />
      </div>
      {filteredSuggestions.length > 0 && draft && (
        <div className="flex flex-wrap gap-1.5">
          {filteredSuggestions.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => commit(s)}
              className="inline-flex items-center gap-1 rounded-full border border-border/60 px-2 py-0.5 text-[11px] text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground"
            >
              <Plus className="size-2.5" />
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
