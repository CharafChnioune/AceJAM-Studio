import * as React from "react";
import { motion } from "framer-motion";
import { Upload, Music, X, Loader2 } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { uploadFile } from "@/lib/api";
import { WaveformPlayer } from "@/components/audio/WaveformPlayer";
import { toast } from "@/components/ui/sonner";
import { cn, formatDuration } from "@/lib/utils";

export interface SourceAudioValue {
  uploadId: string;
  filename: string;
  audioUrl: string;
  duration?: number;
}

interface SourceAudioStepProps {
  value?: SourceAudioValue;
  onChange: (value: SourceAudioValue | undefined) => void;
  description?: string;
  className?: string;
}

export function SourceAudioStep({
  value,
  onChange,
  description,
  className,
}: SourceAudioStepProps) {
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [drag, setDrag] = React.useState(false);

  const upload = useMutation({
    mutationFn: (file: File) => uploadFile(file),
    onSuccess: (resp, file) => {
      if (!resp.success || !resp.upload_id) {
        toast.error(resp.error || "Upload mislukt");
        return;
      }
      const audioUrl = resp.url || URL.createObjectURL(file);
      onChange({
        uploadId: resp.upload_id,
        filename: file.name,
        audioUrl,
      });
      toast.success(`${file.name} geüpload.`);
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const handleFile = (file?: File) => {
    if (!file) return;
    if (!file.type.startsWith("audio/") && !/\.(wav|mp3|flac|ogg|m4a)$/i.test(file.name)) {
      toast.error("Alleen audio-bestanden (wav/mp3/flac/ogg/m4a)");
      return;
    }
    upload.mutate(file);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDrag(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  return (
    <div className={cn("space-y-4", className)}>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0] ?? undefined)}
      />

      {!value ? (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          className={cn(
            "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed bg-card/30 p-12 text-center transition-colors hover:border-primary/40 hover:bg-card/50",
            drag && "border-primary bg-primary/5",
          )}
        >
          {upload.isPending ? (
            <>
              <Loader2 className="size-8 animate-spin text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Uploaden…</p>
            </>
          ) : (
            <>
              <div className="flex size-12 items-center justify-center rounded-full bg-primary/15 text-primary">
                <Upload className="size-5" />
              </div>
              <div className="space-y-1">
                <p className="font-medium">Sleep een audio-bestand hierheen</p>
                <p className="text-xs text-muted-foreground">
                  …of klik om te bladeren — wav, mp3, flac, ogg of m4a
                </p>
              </div>
              {description && (
                <p className="max-w-md text-xs text-muted-foreground">{description}</p>
              )}
            </>
          )}
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-3"
        >
          <div className="flex items-center justify-between gap-2 rounded-md border bg-card/40 px-3 py-2">
            <div className="flex min-w-0 items-center gap-2 text-sm">
              <Music className="size-4 shrink-0 text-primary" />
              <span className="truncate">{value.filename}</span>
              {value.duration && (
                <span className="font-mono text-xs text-muted-foreground">
                  {formatDuration(value.duration)}
                </span>
              )}
            </div>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => onChange(undefined)}
              title="Verwijder"
            >
              <X className="size-3.5" />
            </Button>
          </div>
          <WaveformPlayer
            src={value.audioUrl}
            title={value.filename}
            onReady={(d) =>
              onChange({ ...value, duration: d })
            }
          />
        </motion.div>
      )}

      <div className="text-xs text-muted-foreground">
        <Label className="mb-1 block text-xs uppercase tracking-wider text-muted-foreground">
          Upload-id
        </Label>
        <code className="rounded bg-background/40 px-2 py-1 font-mono text-[11px]">
          {value?.uploadId || "—"}
        </code>
      </div>
    </div>
  );
}
