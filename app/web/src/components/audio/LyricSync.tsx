import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { Mic2 } from "lucide-react";
import { api } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface LyricSyncProps {
  resultId?: string;
  uploadId?: string;
  audioCurrentTime: number;
  staticLyrics?: string;
  className?: string;
}

interface LRCLine {
  time: number;
  text: string;
}

function parseLRC(raw: unknown): LRCLine[] {
  const out: LRCLine[] = [];
  if (raw === null || raw === undefined) return out;
  const source = Array.isArray(raw) ? raw.join("\n") : String(raw);
  if (!source) return out;
  const lines = source.split(/\r?\n/);
  const re = /^\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]/;
  for (const line of lines) {
    const m = line.match(re);
    if (!m) continue;
    const min = Number(m[1]);
    const sec = Number(m[2]);
    const ms = m[3] ? Number(m[3].padEnd(3, "0")) : 0;
    const text = line.replace(re, "").trim();
    if (text) out.push({ time: min * 60 + sec + ms / 1000, text });
  }
  return out.sort((a, b) => a.time - b.time);
}

export function LyricSync({
  resultId,
  uploadId,
  audioCurrentTime,
  staticLyrics,
  className,
}: LyricSyncProps) {
  const lrcQuery = useQuery({
    queryKey: ["lrc", resultId, uploadId],
    queryFn: () =>
      api.post<{ success: boolean; lrc?: string; error?: string }>("/api/lrc", {
        result_id: resultId,
        upload_id: uploadId,
      }),
    enabled: !!(resultId || uploadId),
    staleTime: 60_000,
  });

  const lines = React.useMemo(
    () => parseLRC(lrcQuery.data?.lrc ?? ""),
    [lrcQuery.data?.lrc],
  );

  const activeIndex = React.useMemo(() => {
    if (!lines.length) return -1;
    let lo = 0;
    let hi = lines.length - 1;
    let res = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (lines[mid].time <= audioCurrentTime) {
        res = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return res;
  }, [lines, audioCurrentTime]);

  const containerRef = React.useRef<HTMLDivElement>(null);
  const activeRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!activeRef.current || !containerRef.current) return;
    activeRef.current.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [activeIndex]);

  if (lrcQuery.isLoading) {
    return (
      <div className={cn("space-y-2 rounded-xl border bg-card/40 p-4", className)}>
        <Skeleton className="h-3 w-24" />
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-4 w-full" />
        ))}
      </div>
    );
  }

  if (!lines.length) {
    if (!staticLyrics) return null;
    return (
      <div className={cn("rounded-xl border bg-card/40 p-4", className)}>
        <div className="mb-2 flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-muted-foreground">
          <Mic2 className="size-3" /> Lyrics
        </div>
        <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-muted-foreground">
          {staticLyrics}
        </pre>
      </div>
    );
  }

  return (
    <div className={cn("rounded-xl border bg-card/40 p-4", className)}>
      <div className="mb-2 flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-muted-foreground">
        <Mic2 className="size-3" /> Karaoke
      </div>
      <div
        ref={containerRef}
        className="scroll-fade-y no-scrollbar relative max-h-48 overflow-y-auto"
      >
        <div className="space-y-1.5 py-12 text-center">
          <AnimatePresence initial={false}>
            {lines.map((l, i) => {
              const active = i === activeIndex;
              const past = i < activeIndex;
              return (
                <motion.div
                  key={`${l.time}-${i}`}
                  ref={active ? activeRef : null}
                  initial={{ opacity: 0.4 }}
                  animate={{
                    opacity: active ? 1 : past ? 0.35 : 0.55,
                    scale: active ? 1.02 : 1,
                  }}
                  transition={{ duration: 0.18 }}
                  className={cn(
                    "px-4 text-base leading-relaxed transition-colors",
                    active && "font-display font-semibold text-primary",
                    !active && past && "text-muted-foreground/60",
                    !active && !past && "text-foreground/70",
                  )}
                >
                  {l.text}
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
