import * as React from "react";
import { Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { Music2, Search, Trash2, Download, Video } from "lucide-react";

import { getMlxVideoAttachments, listCommunity, deleteSong, type MlxVideoAttachment, type SongMeta } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { WaveformPlayer } from "@/components/audio/WaveformPlayer";
import { LyricSync } from "@/components/audio/LyricSync";
import { MfluxArtMaker } from "@/components/mflux/MfluxArtMaker";
import { toast } from "@/components/ui/sonner";
import { cn, formatDuration } from "@/lib/utils";

function ActiveAudio({ active }: { active: SongMeta }) {
  const [time, setTime] = React.useState(0);
  return (
    <div className="space-y-3">
      <WaveformPlayer
        src={active.audio_url!}
        title={active.title}
        artist={active.artist_name}
        onTimeUpdate={setTime}
      />
      <LyricSync
        resultId={active.result_id || active.song_id}
        audioCurrentTime={time}
        staticLyrics={active.lyrics}
      />
    </div>
  );
}

function songTags(s: SongMeta): string[] {
  if (Array.isArray(s.tags)) return s.tags;
  if (typeof s.tags === "string") return s.tags.split(",").map((t) => t.trim()).filter(Boolean);
  return [];
}

function songId(s: SongMeta): string {
  return s.song_id || s.result_id || `${s.title}-${s.created_at}`;
}

export function Library() {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["library", "community"], queryFn: () => listCommunity() });
  const videoAttachmentsQ = useQuery({
    queryKey: ["mlx-video", "attachments"],
    queryFn: () => getMlxVideoAttachments(),
    staleTime: 15_000,
  });
  const songs = q.data?.songs ?? [];
  const videoAttachments = videoAttachmentsQ.data?.attachments ?? [];

  const [search, setSearch] = React.useState("");
  const [language, setLanguage] = React.useState("__all__");
  const [model, setModel] = React.useState("__all__");
  const [active, setActive] = React.useState<SongMeta | null>(null);

  const filtered = React.useMemo(() => {
    const s = search.trim().toLowerCase();
    return songs.filter((song) => {
      if (language !== "__all__" && (song.vocal_language ?? "") !== language) return false;
      if (model !== "__all__" && (song.song_model ?? "") !== model) return false;
      if (!s) return true;
      const blob = [
        song.title,
        song.artist_name,
        song.caption,
        song.song_model,
        ...(songTags(song)),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return blob.includes(s);
    });
  }, [songs, search, language, model]);

  const allLanguages = Array.from(new Set(songs.map((s) => s.vocal_language).filter(Boolean))) as string[];
  const allModels = Array.from(new Set(songs.map((s) => s.song_model).filter(Boolean))) as string[];
  const videosForSong = React.useCallback((song: SongMeta): MlxVideoAttachment[] => {
    const ids = new Set(
      [song.song_id, song.result_id, songId(song)]
        .filter(Boolean)
        .map((item) => String(item)),
    );
    return videoAttachments.filter((item) => ids.has(String(item.target_id || "")));
  }, [videoAttachments]);

  const remove = useMutation({
    mutationFn: (id: string) => deleteSong(id),
    onSuccess: (resp) => {
      if (!resp.success) {
        toast.error(resp.error || "Verwijderen mislukt");
        return;
      }
      toast.success("Track verwijderd.");
      setActive(null);
      qc.invalidateQueries({ queryKey: ["library"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6 px-6 py-10 sm:px-10">
      <header className="space-y-2">
        <h1 className="font-display text-3xl font-semibold">Library</h1>
        <p className="text-sm text-muted-foreground">
          {songs.length} tracks · klik een track voor playback, artwork en acties.
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[240px]">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Zoek op titel, artiest, tag…"
            className="pl-9"
          />
        </div>
        <Select value={language} onValueChange={setLanguage}>
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="Taal" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__all__">Alle talen</SelectItem>
            {allLanguages.map((l) => (
              <SelectItem key={l} value={l}>{l}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={model} onValueChange={setModel}>
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder="Model" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__all__">Alle modellen</SelectItem>
            {allModels.map((m) => (
              <SelectItem key={m} value={m}>{m}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {q.isLoading && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-32 rounded-xl" />
          ))}
        </div>
      )}

      {!q.isLoading && filtered.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center gap-3 py-12 text-center text-muted-foreground">
            <Music2 className="size-8 opacity-60" />
            <p>
              {songs.length === 0
                ? "Nog geen tracks. Start een wizard om je eerste te maken."
                : "Geen resultaten voor deze filters."}
            </p>
          </CardContent>
        </Card>
      )}

      <motion.div
        initial="hidden"
        animate="show"
        variants={{
          hidden: {},
          show: { transition: { staggerChildren: 0.03 } },
        }}
        className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3"
      >
        <AnimatePresence>
          {filtered.map((s) => {
            const linkedVideos = videosForSong(s);
            return (
            <motion.div
              key={songId(s)}
              layout
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              variants={{ show: { opacity: 1, y: 0 } }}
            >
              <button
                type="button"
                onClick={() => setActive(s)}
                className="group flex w-full gap-3 rounded-xl border bg-card/50 p-3 text-left transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-md"
              >
                <div className="aspect-square size-20 shrink-0 overflow-hidden rounded-lg bg-muted/40">
                  {s.art?.url ? (
                    <img src={s.art.url} alt="cover" className="size-full object-cover transition-transform group-hover:scale-105" />
                  ) : (
                    <div className="flex size-full items-center justify-center text-muted-foreground/60">
                      <Music2 className="size-5" />
                    </div>
                  )}
                </div>
                <div className="min-w-0 flex-1 space-y-1">
                  <p className="truncate font-display text-sm font-semibold">{s.title || "Untitled"}</p>
                  <p className="truncate text-xs text-muted-foreground">{s.artist_name || "—"}</p>
                  <div className="flex flex-wrap gap-1">
                    {songTags(s).slice(0, 3).map((t) => (
                      <Badge key={t} variant="muted" className="text-[10px]">{t}</Badge>
                    ))}
                    {linkedVideos.length > 0 && (
                      <Badge variant="outline" className="gap-1 text-[10px]">
                        <Video className="size-3" /> {linkedVideos.length}
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2 pt-1 text-[10px] text-muted-foreground">
                    {s.duration && <span>{formatDuration(s.duration)}</span>}
                    {s.song_model && <span>· {s.song_model}</span>}
                    {s.vocal_language && <span>· {s.vocal_language}</span>}
                  </div>
                </div>
              </button>
            </motion.div>
            );
          })}
        </AnimatePresence>
      </motion.div>

      <Dialog open={!!active} onOpenChange={(o) => !o && setActive(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>{active?.title || "Untitled"}</DialogTitle>
            <DialogDescription>{active?.artist_name || "—"}</DialogDescription>
          </DialogHeader>
          {active?.audio_url && (
            <ActiveAudio active={active} />
          )}
          {active && (
            <div className="space-y-3">
              <MfluxArtMaker
                title={active.title}
                artist={active.artist_name}
                context={String(active.caption || active.tags || "")}
                targetType="song"
                targetId={active.song_id || active.result_id}
                compact
              />
              <div className="grid grid-cols-3 gap-2">
                {active.bpm && (
                  <div className="rounded-md border bg-card/40 p-2 text-center">
                    <p className="text-[10px] uppercase tracking-widest text-muted-foreground">BPM</p>
                    <p className="font-mono text-sm">{active.bpm}</p>
                  </div>
                )}
                {active.key_scale && (
                  <div className="rounded-md border bg-card/40 p-2 text-center">
                    <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Key</p>
                    <p className="font-mono text-sm">{active.key_scale}</p>
                  </div>
                )}
                {active.duration && (
                  <div className="rounded-md border bg-card/40 p-2 text-center">
                    <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Duur</p>
                    <p className="font-mono text-sm">{formatDuration(active.duration)}</p>
                  </div>
                )}
              </div>
              <div className="flex flex-wrap gap-1">
                {songTags(active).map((t) => (
                  <Badge key={t} variant="muted" className="text-[10px]">{t}</Badge>
                ))}
              </div>
              {videosForSong(active).length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Linked videos</p>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {videosForSong(active).map((video) => (
                      <div key={`${video.result_id}-${video.target_id}`} className="overflow-hidden rounded-md border bg-card/40">
                        <video
                          src={video.url || video.video_url}
                          poster={video.poster_url || undefined}
                          controls
                          className="aspect-video w-full bg-black object-contain"
                        />
                        <div className="space-y-1 p-2">
                          <p className="truncate text-xs font-medium">{video.model_label || "MLX video"}</p>
                          <p className="line-clamp-2 text-[10px] text-muted-foreground">{video.prompt}</p>
                          {(video.url || video.video_url) && (
                            <Button asChild variant="ghost" size="sm" className="h-7 px-2 text-xs">
                              <a href={video.url || video.video_url} download>Download MP4</a>
                            </Button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div className="flex flex-wrap gap-2">
                {active.audio_url && (
                  <Button asChild variant="outline" size="sm" className="gap-2">
                    <a href={active.audio_url} download target="_blank" rel="noreferrer">
                      <Download className="size-3.5" /> Download
                    </a>
                  </Button>
                )}
                {active.audio_url && (
                  <Button asChild variant="outline" size="sm" className="gap-2">
                    <Link
                      to="/wizard/video"
                      state={{
                        audio_url: active.audio_url,
                        title: active.title,
                        artist_name: active.artist_name,
                        prompt: String(active.caption || active.tags || active.title || ""),
                        target_type: "song",
                        target_id: active.song_id || active.result_id || songId(active),
                      }}
                    >
                      <Video className="size-3.5" /> Create video
                    </Link>
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    if (!active.song_id && !active.result_id) return;
                    if (confirm("Track verwijderen?")) {
                      remove.mutate(active.song_id || active.result_id || "");
                    }
                  }}
                  className="gap-2 text-destructive hover:text-destructive"
                  disabled={remove.isPending}
                >
                  <Trash2 className="size-3.5" /> Verwijder
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
