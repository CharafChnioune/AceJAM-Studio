import * as React from "react";
import WaveSurfer from "wavesurfer.js";
import { Gauge, Play, Pause, Volume2, VolumeX, Repeat, Download, Rewind, FastForward } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn, formatDuration } from "@/lib/utils";

export interface WaveformMetadata {
  model?: unknown;
  quality?: unknown;
  duration?: unknown;
  bpm?: unknown;
  key?: unknown;
  seed?: unknown;
  resultId?: unknown;
}

interface WaveformPlayerProps {
  src: string;
  title?: string;
  artist?: string;
  metadata?: WaveformMetadata;
  actions?: React.ReactNode;
  className?: string;
  onReady?: (duration: number) => void;
  onRegionChange?: (start: number, end: number) => void;
  onTimeUpdate?: (time: number) => void;
}

export function WaveformPlayer({
  src,
  title,
  artist,
  metadata,
  actions,
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
  const [playbackRate, setPlaybackRate] = React.useState("1");

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
      height: 108,
      normalize: true,
      url: src,
    });
    wsRef.current = ws;

    ws.on("ready", () => {
      const d = ws.getDuration();
      setDuration(d);
      setLoaded(true);
      ws.setVolume(volume);
      ws.setPlaybackRate(Number(playbackRate) || 1);
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

  React.useEffect(() => {
    wsRef.current?.setPlaybackRate(Number(playbackRate) || 1);
  }, [playbackRate]);

  const togglePlay = () => wsRef.current?.playPause();
  const seekBy = (delta: number) => {
    const ws = wsRef.current;
    if (!ws) return;
    ws.setTime(Math.max(0, Math.min(duration || ws.getDuration(), currentTime + delta)));
  };

  const chips = [
    metadata?.model ? ["Model", metadata.model] : null,
    metadata?.quality ? ["Quality", metadata.quality] : null,
    metadata?.duration ? ["Duration", typeof metadata.duration === "number" ? formatDuration(Number(metadata.duration)) : metadata.duration] : null,
    metadata?.bpm ? ["BPM", metadata.bpm] : null,
    metadata?.key ? ["Key", metadata.key] : null,
    metadata?.seed !== undefined && metadata.seed !== "" ? ["Seed", metadata.seed] : null,
    metadata?.resultId ? ["Result", metadata.resultId] : null,
  ].filter(Boolean) as Array<[string, unknown]>;

  return (
    <div className={cn("rounded-xl border bg-card/70 p-4 shadow-sm", className)}>
      <div className="flex flex-wrap items-start justify-between gap-3 pb-3">
        <div className="min-w-0 flex-1">
          {title && (
            <p className="truncate font-display text-base font-semibold leading-tight">
              {title}
            </p>
          )}
          {artist && (
            <p className="truncate text-xs text-muted-foreground">{artist}</p>
          )}
          {chips.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {chips.slice(0, 7).map(([label, value]) => (
                <Badge key={label} variant="outline" className="max-w-[220px] gap-1 truncate text-[10px]">
                  <span className="text-muted-foreground">{label}</span>
                  <span className="truncate">{String(value)}</span>
                </Badge>
              ))}
            </div>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {actions}
          <Button variant="ghost" size="icon-sm" asChild title="Download">
            <a href={src} download target="_blank" rel="noreferrer">
              <Download className="size-4" />
            </a>
          </Button>
        </div>
      </div>

      <div ref={containerRef} className={cn(!loaded && "min-h-[108px] animate-pulse rounded-md bg-muted/20")} />

      <div className="mt-2 flex items-center justify-between gap-3 font-mono text-xs tabular-nums text-muted-foreground">
        <span>{formatDuration(currentTime)}</span>
        <span>{formatDuration(duration)}</span>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="icon-sm"
          onClick={() => seekBy(-10)}
          disabled={!loaded}
          title="10 seconden terug"
        >
          <Rewind className="size-3.5" />
        </Button>
        <Button
          variant="default"
          size="icon"
          onClick={togglePlay}
          disabled={!loaded}
          className="rounded-full"
        >
          {isPlaying ? <Pause className="size-4" /> : <Play className="size-4" />}
        </Button>
        <Button
          variant="outline"
          size="icon-sm"
          onClick={() => seekBy(10)}
          disabled={!loaded}
          title="10 seconden vooruit"
        >
          <FastForward className="size-3.5" />
        </Button>
        <div className="ml-auto flex flex-wrap items-center gap-2">
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
          <div className="flex items-center gap-1.5">
            <Gauge className="size-3.5 text-muted-foreground" />
            <Select value={playbackRate} onValueChange={setPlaybackRate}>
              <SelectTrigger className="h-8 w-[86px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0.75">0.75x</SelectItem>
                <SelectItem value="1">1x</SelectItem>
                <SelectItem value="1.25">1.25x</SelectItem>
                <SelectItem value="1.5">1.5x</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>
    </div>
  );
}
