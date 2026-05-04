import { Badge } from "@/components/ui/badge";
import { formatDuration } from "@/lib/utils";

function text(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function wordCount(value: unknown): number {
  const raw = String(value || "").trim();
  if (!raw || raw === "[Instrumental]") return 0;
  return raw.split(/\s+/).filter(Boolean).length;
}

export function RenderInsightPanel({
  payload,
  warnings = [],
}: {
  payload: Record<string, unknown>;
  warnings?: string[];
}) {
  const duration = Number(payload.duration || payload.audio_duration || 0);
  const lyricsWords = wordCount(payload.lyrics);
  const instrumental = payload.instrumental === true || String(payload.lyrics || "").trim() === "[Instrumental]";
  const sourceMode = payload.src_audio_id || payload.src_result_id || payload.audio_code_string ? "source audio" : "text to music";
  const loraName = payload.use_lora ? text(payload.lora_adapter_name || payload.lora_adapter_path, "") : "";
  const chips = [
    ["Mode", text(payload.task_type || sourceMode)],
    ["Model", text(payload.song_model || "auto")],
    ...(loraName ? ([["LoRA", loraName]] as Array<[string, string]>) : []),
    ["Quality", text(payload.quality_profile || "auto")],
    ["Duration", duration ? formatDuration(duration) : "auto"],
    ["Lyrics", instrumental ? "instrumental" : `${lyricsWords} words`],
    ["Seed", text(payload.seed || "random")],
  ];

  return (
    <div className="rounded-xl border bg-card/45 p-4">
      <div className="flex flex-wrap gap-2">
        {chips.map(([label, value]) => (
          <Badge key={label} variant="outline" className="gap-1.5">
            <span className="text-muted-foreground">{label}</span>
            <span>{String(value)}</span>
          </Badge>
        ))}
        {Boolean(payload.bpm || payload.key_scale) && (
          <Badge variant="secondary">
            {text(payload.bpm || "auto")} BPM · {text(payload.key_scale || "auto")}
          </Badge>
        )}
      </div>
      {warnings.length > 0 && (
        <div className="mt-3 space-y-1 text-xs text-muted-foreground">
          {warnings.map((warning, index) => (
            <p key={`${warning}-${index}`}>{warning}</p>
          ))}
        </div>
      )}
    </div>
  );
}
