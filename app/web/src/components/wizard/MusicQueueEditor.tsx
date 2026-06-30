import { ArrowDown, ArrowUp, Copy, ListMusic, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { QueueSummaryItem } from "@/lib/musicQueue";

interface MusicQueueEditorProps {
  items: QueueSummaryItem[];
  activeIndex?: number;
  onSelect?: (index: number) => void;
  onRemove: (index: number) => void;
  onDuplicate: (index: number) => void;
  onMove: (index: number, direction: -1 | 1) => void;
  className?: string;
}

export function MusicQueueEditor({
  items,
  activeIndex = -1,
  onSelect,
  onRemove,
  onDuplicate,
  onMove,
  className,
}: MusicQueueEditorProps) {
  return (
    <div className={cn("space-y-3 rounded-xl border bg-card/40 p-4", className)}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <ListMusic className="size-4" />
          Queue
        </div>
        <span className="text-xs text-muted-foreground">{items.length} klaar gezet</span>
      </div>
      {items.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          Nog niets in de queue. Werk eerst het huidige item af en kies dan <strong>Nog een toevoegen</strong>.
        </p>
      ) : (
        <div className="space-y-2">
          {items.map((item, index) => (
            <div
              key={`${item.title}-${index}`}
              className={cn(
                "rounded-lg border border-border/60 bg-background/40 p-3",
                index === activeIndex && "border-primary/50 bg-primary/10",
              )}
            >
              <div className="flex items-start justify-between gap-3">
                <button
                  type="button"
                  onClick={() => onSelect?.(index)}
                  className="min-w-0 flex-1 text-left"
                >
                  <p className="truncate text-sm font-medium">{item.title}</p>
                  {(item.subtitle || item.detail) && (
                    <p className="mt-0.5 truncate text-xs text-muted-foreground">
                      {[item.subtitle, item.detail].filter(Boolean).join(" · ")}
                    </p>
                  )}
                </button>
                <div className="flex items-center gap-1">
                  <Button type="button" size="icon" variant="ghost" onClick={() => onMove(index, -1)} disabled={index === 0}>
                    <ArrowUp className="size-4" />
                  </Button>
                  <Button
                    type="button"
                    size="icon"
                    variant="ghost"
                    onClick={() => onMove(index, 1)}
                    disabled={index >= items.length - 1}
                  >
                    <ArrowDown className="size-4" />
                  </Button>
                  <Button type="button" size="icon" variant="ghost" onClick={() => onDuplicate(index)}>
                    <Copy className="size-4" />
                  </Button>
                  <Button type="button" size="icon" variant="ghost" onClick={() => onRemove(index)}>
                    <Trash2 className="size-4" />
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
