import { ExternalLink, Loader2, Music4 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { GenerationAudioList } from "@/components/wizard/GenerationAudioList";
import type { GenerationJob } from "@/lib/api";

function text(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

export function GenerationJobStatus({
  job,
  jobId,
  onOpen,
}: {
  job: GenerationJob | null;
  jobId?: string;
  onOpen: () => void;
}) {
  const state = text(job?.state, jobId ? "queued" : "idle");
  const progress = Math.max(0, Math.min(100, Number(job?.progress ?? 0) || 0));
  const summary = job?.payload_summary ?? {};
  const result = job?.result ?? null;
  const completedCount = Array.isArray(result?.audios) ? result.audios.length : 0;

  return (
    <div className="rounded-xl border border-primary/30 bg-primary/10 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
            <Loader2 className="size-4 animate-spin" />
          </div>
          <div className="min-w-0">
            <p className="font-medium">Render draait op de achtergrond</p>
            <p className="mt-1 text-xs text-muted-foreground">
              {text(job?.stage || job?.status, "Queued")} · {text(jobId || job?.id)}
            </p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              <Badge variant="outline">{state}</Badge>
              {Boolean(summary.song_model) && <Badge variant="secondary">{text(summary.song_model)}</Badge>}
              {Boolean(summary.duration) && <Badge variant="muted">{text(summary.duration)}s</Badge>}
              {completedCount > 0 && <Badge variant="secondary">{completedCount} take{completedCount === 1 ? "" : "s"} klaar</Badge>}
            </div>
          </div>
        </div>
        <Button variant="outline" size="sm" onClick={onOpen} disabled={!jobId && !job?.id}>
          <ExternalLink className="size-3.5" />
          Open job
        </Button>
      </div>
      <Progress value={progress} className="mt-4 h-1.5" />
      <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
        <Music4 className="size-3.5" />
        Complete takes verschijnen hier meteen; de volgende takes renderen door op de achtergrond.
      </div>
      {result && completedCount > 0 && (
        <GenerationAudioList
          result={result}
          title={text(summary.title, "Song render")}
          artist={text(summary.artist_name, "")}
          className="mt-4 space-y-3"
        />
      )}
    </div>
  );
}
