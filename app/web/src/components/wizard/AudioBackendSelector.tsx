import { Label } from "@/components/ui/label";
import { DEFAULT_AUDIO_BACKEND, normalizeAudioBackend, type AudioBackend } from "@/lib/audioBackend";

export function AudioBackendSelector({
  value,
  onChange,
}: {
  value?: string;
  onChange: (value: AudioBackend) => void;
}) {
  const normalized = normalizeAudioBackend(value || DEFAULT_AUDIO_BACKEND);
  if (normalized !== DEFAULT_AUDIO_BACKEND) onChange(DEFAULT_AUDIO_BACKEND);
  return (
    <div className="space-y-1.5">
      <Label>Audio backend</Label>
      <div className="flex h-10 items-center rounded-md border bg-muted/40 px-3 text-sm font-medium">
        MLX
      </div>
      <p className="text-xs text-muted-foreground">
        MLX is de enige audio-runtime in deze UI en wordt voor alle ACE-Step renders gebruikt.
      </p>
    </div>
  );
}
