import * as React from "react";
import WaveSurfer from "wavesurfer.js";
import { Play, Pause, Volume2, VolumeX, Repeat, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { cn, formatDuration } from "@/lib/utils";

interface WaveformPlayerProps {
  src: string;
  title?: string;
  artist?: string;
  className?: string;
  onReady?: (duration: number) => void;
  onRegionChange?: (start: number, end: number) => void;
  onTimeUpdate?: (time: number) => void;
}

export function WaveformPlayer({
  src,
  title,
  artist,
  className,
  onReady,
  onTimeUpdate,
}: WaveformPlayerProps) {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const wsRef = React.useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = React.useState(false);
  const [currentTime, setCurrentTime] = React.useState(0);
  const [duration, setDuration] = React.useState(0);
  const [volume, setVolume] = React.useState(0.9);
  const [muted, setMuted] = React.useState(false);
  const [loop, setLoop] = React.useState(false);
  const [loaded, setLoaded] = React.useState(false);

  React.useEffect(() => {
    if (!containerRef.current || !src) return;
    setLoaded(false);
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "rgba(255,255,255,0.18)",
      progressColor: "oklch(0.85 0.18 165 / 0.95)",
      cursorColor: "rgba(255,255,255,0.6)",
      barWidth: 2,
      barRadius: 2,
      barGap: 2,
      height: 88,
      normalize: true,
      url: src,
    });
    wsRef.current = ws;

    ws.on("ready", () => {
      const d = ws.getDuration();
      setDuration(d);
      setLoaded(true);
      ws.setVolume(volume);
      onReady?.(d);
    });
    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => {
      if (loop) {
        ws.setTime(0);
        ws.play();
      } else {
        setIsPlaying(false);
      }
    });
    ws.on("timeupdate", (t) => {
      setCurrentTime(t);
      onTimeUpdate?.(t);
    });

    return () => {
      ws.destroy();
      wsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);

  React.useEffect(() => {
    wsRef.current?.setVolume(muted ? 0 : volume);
  }, [volume, muted]);

  const togglePlay = () => wsRef.current?.playPause();

  return (
    <div className={cn("rounded-xl border bg-card/60 p-4", className)}>
      <div className="flex items-start justify-between gap-3 pb-3">
        <div className="min-w-0">
          {title && (
            <p className="truncate font-display text-base font-semibold leading-tight">
              {title}
            </p>
          )}
          {artist && (
            <p className="truncate text-xs text-muted-foreground">{artist}</p>
          )}
        </div>
        <a
          href={src}
          download
          target="_blank"
          rel="noreferrer"
          className="text-muted-foreground transition-colors hover:text-foreground"
          title="Download"
        >
          <Download className="size-4" />
        </a>
      </div>

      <div ref={containerRef} className={cn(!loaded && "min-h-[88px] animate-pulse rounded-md bg-muted/20")} />

      <div className="mt-3 flex items-center gap-3">
        <Button
          variant="default"
          size="icon"
          onClick={togglePlay}
          disabled={!loaded}
          className="rounded-full"
        >
          {isPlaying ? <Pause className="size-4" /> : <Play className="size-4" />}
        </Button>
        <span className="font-mono text-xs tabular-nums text-muted-foreground">
          {formatDuration(currentTime)} / {formatDuration(duration)}
        </span>
        <div className="ml-auto flex items-center gap-2">
          <Button
            variant={loop ? "secondary" : "ghost"}
            size="icon-sm"
            onClick={() => setLoop((v) => !v)}
            title={loop ? "Loop aan" : "Loop uit"}
          >
            <Repeat className="size-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setMuted((m) => !m)}
            title={muted ? "Unmute" : "Mute"}
          >
            {muted ? <VolumeX className="size-3.5" /> : <Volume2 className="size-3.5" />}
          </Button>
          <Slider
            value={[muted ? 0 : volume]}
            min={0}
            max={1}
            step={0.01}
            onValueChange={(v) => {
              setMuted(false);
              setVolume(v[0] ?? 0);
            }}
            className="w-24"
          />
        </div>
      </div>
    </div>
  );
}
