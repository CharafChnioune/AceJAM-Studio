import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { DEFAULT_AUDIO_BACKEND, normalizeAudioBackend, type AudioBackend } from "@/lib/audioBackend";

export function AudioBackendSelector({
  value,
  onChange,
}: {
  value?: string;
  onChange: (value: AudioBackend) => void;
}) {
  const normalized = normalizeAudioBackend(value || DEFAULT_AUDIO_BACKEND);
  return (
    <div className="space-y-1.5">
      <Label>Audio backend</Label>
      <Select value={normalized} onValueChange={(next) => onChange(normalizeAudioBackend(next))}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="mlx">MLX (aanbevolen)</SelectItem>
          <SelectItem value="mps_torch">MPS/Torch</SelectItem>
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        MLX is standaard; MPS/Torch blijft beschikbaar als fallback.
      </p>
    </div>
  );
}
