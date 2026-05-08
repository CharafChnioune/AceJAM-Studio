import * as React from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { Download, HardDrive, Image as ImageIcon, Library as LibraryIcon, Music2, Search, Trash2, Video } from "lucide-react";

import {
  deleteLibraryItem,
  getMlxVideoAttachments,
  listLibrary,
  type LibraryItem,
  type MlxVideoAttachment,
} from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { WaveformPlayer } from "@/components/audio/WaveformPlayer";
import { LyricSync } from "@/components/audio/LyricSync";
import { MfluxArtMaker } from "@/components/mflux/MfluxArtMaker";
import { toast } from "@/components/ui/sonner";
import { cn, formatDuration } from "@/lib/utils";

type LibraryTab = "all" | "songs" | "results" | "images" | "videos";
type DeleteKind = "song" | "result-audio" | "image" | "video";

function tags(item: LibraryItem): string[] {
  if (Array.isArray(item.tags)) return item.tags.map(String).filter(Boolean);
  if (typeof item.tags === "string") return item.tags.split(",").map((t) => t.trim()).filter(Boolean);
  return [];
}

function itemTitle(item: LibraryItem) {
  if (item.kind === "image") return item.title || "MFLUX image";
  return item.title || (item.kind === "video" ? "MLX video" : "Untitled");
}

function itemArtist(item: LibraryItem) {
  return item.artist_name || "—";
}

function deleteKind(item: LibraryItem): DeleteKind {
  if (item.kind === "video") return "video";
  if (item.kind === "image") return "image";
  return item.source === "song" ? "song" : "result-audio";
}

function sourceLabel(item: LibraryItem) {
  if (item.kind === "video") return "Video";
  if (item.kind === "image") return "Image";
  if (item.source === "song") return "Song";
  return "Result WAV";
}

function mediaUrl(item: LibraryItem) {
  if (item.kind === "video") return item.video_url || item.url || item.download_url || "";
  if (item.kind === "image") return item.image_url || item.thumbnail_url || item.url || item.download_url || "";
  return item.audio_url || item.download_url || "";
}

function ActiveAudio({ active }: { active: LibraryItem }) {
  const [time, setTime] = React.useState(0);
  return (
    <div className="space-y-3">
      <WaveformPlayer
        src={active.audio_url!}
        title={itemTitle(active)}
        artist={itemArtist(active)}
        onTimeUpdate={setTime}
      />
      <LyricSync
        resultId={active.result_id || active.song_id || active.id}
        audioCurrentTime={time}
        staticLyrics={active.lyrics}
      />
    </div>
  );
}

function Stat({ label, value }: { label: string; value?: React.ReactNode }) {
  if (value === undefined || value === null || value === "") return null;
  return (
    <div className="rounded-md border bg-card/40 p-2 text-center">
      <p className="text-[10px] uppercase tracking-widest text-muted-foreground">{label}</p>
      <p className="truncate font-mono text-sm">{value}</p>
    </div>
  );
}

export function Library() {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["library"], queryFn: () => listLibrary() });
  const videoAttachmentsQ = useQuery({
    queryKey: ["mlx-video", "attachments"],
    queryFn: () => getMlxVideoAttachments(),
    staleTime: 15_000,
  });
  const items = q.data?.items ?? [];
  const counts = q.data?.counts ?? { all: 0, songs: 0, results: 0, images: 0, videos: 0, audio: 0 };
  const videoAttachments = videoAttachmentsQ.data?.attachments ?? [];

  const [tab, setTab] = React.useState<LibraryTab>("all");
  const [search, setSearch] = React.useState("");
  const [model, setModel] = React.useState("__all__");
  const [active, setActive] = React.useState<LibraryItem | null>(null);
  const [deleteTarget, setDeleteTarget] = React.useState<LibraryItem | null>(null);
  const [deleteText, setDeleteText] = React.useState("");

  const filtered = React.useMemo(() => {
    const s = search.trim().toLowerCase();
    return items.filter((item) => {
      if (tab === "songs" && item.source !== "song") return false;
      if (tab === "results" && item.source !== "result") return false;
      if (tab === "images" && item.kind !== "image") return false;
      if (tab === "videos" && item.kind !== "video") return false;
      if (model !== "__all__" && String(item.song_model || item.model_label || "") !== model) return false;
      if (!s) return true;
      const blob = [
        itemTitle(item),
        itemArtist(item),
        item.caption,
        item.prompt,
        item.song_model,
        item.model_label,
        sourceLabel(item),
        ...tags(item),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return blob.includes(s);
    });
  }, [items, search, tab, model]);

  const allModels = Array.from(new Set(items.map((item) => item.song_model || item.model_label).filter(Boolean))) as string[];
  const videosForItem = React.useCallback((item: LibraryItem): MlxVideoAttachment[] => {
    const ids = new Set(
      [item.song_id, item.result_id, item.id]
        .filter(Boolean)
        .map((value) => String(value)),
    );
    return videoAttachments.filter((video) => ids.has(String(video.target_id || "")));
  }, [videoAttachments]);

  const remove = useMutation({
    mutationFn: (item: LibraryItem) =>
      deleteLibraryItem({
        id: item.id,
        kind: deleteKind(item),
        result_id: item.result_id,
        song_id: item.song_id,
        audio_id: item.audio_id,
        filename: item.filename,
        confirm: "DELETE",
      }),
    onSuccess: (resp) => {
      if (!resp.success) {
        toast.error(resp.error || "Verwijderen mislukt");
        return;
      }
      toast.success("Media van schijf verwijderd.");
      setActive(null);
      setDeleteTarget(null);
      setDeleteText("");
      qc.invalidateQueries({ queryKey: ["library"] });
      qc.invalidateQueries({ queryKey: ["mlx-video"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const activeVideos = active ? videosForItem(active) : [];

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6 px-6 py-10 sm:px-10">
      <header className="space-y-2">
        <h1 className="font-display text-3xl font-semibold">Library</h1>
        <p className="text-sm text-muted-foreground">
          {counts.audio} audio · {counts.images} images · {counts.videos} videos · direct bekijken, afspelen, downloaden of permanent van schijf verwijderen.
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-2">
        <Tabs value={tab} onValueChange={(value) => setTab(value as LibraryTab)}>
          <TabsList>
            <TabsTrigger value="all">All {counts.all}</TabsTrigger>
            <TabsTrigger value="songs">Songs {counts.songs}</TabsTrigger>
            <TabsTrigger value="results">Results {counts.results}</TabsTrigger>
            <TabsTrigger value="images">Images {counts.images}</TabsTrigger>
            <TabsTrigger value="videos">Videos {counts.videos}</TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="relative min-w-[240px] flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Zoek titel, artiest, tag, model..."
            className="pl-9"
          />
        </div>
        <Select value={model} onValueChange={setModel}>
          <SelectTrigger className="w-[220px]">
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
            <LibraryIcon className="size-8 opacity-60" />
            <p>
              {items.length === 0
                ? `Nog geen media gevonden. Songs: ${counts.songs}, results: ${counts.results}, images: ${counts.images}, videos: ${counts.videos}.`
                : "Geen resultaten voor deze filters."}
            </p>
          </CardContent>
        </Card>
      )}

      <motion.div
        initial="hidden"
        animate="show"
        variants={{ hidden: {}, show: { transition: { staggerChildren: 0.03 } } }}
        className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3"
      >
        <AnimatePresence>
          {filtered.map((item) => {
            const linkedVideos = videosForItem(item);
            const isVideo = item.kind === "video";
            const isImage = item.kind === "image";
            return (
              <motion.div
                key={item.id}
                layout
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                variants={{ show: { opacity: 1, y: 0 } }}
              >
                <button
                  type="button"
                  onClick={() => setActive(item)}
                  className="group flex w-full gap-3 rounded-xl border bg-card/50 p-3 text-left transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-md"
                >
                  <div className="aspect-square size-20 shrink-0 overflow-hidden rounded-lg bg-muted/40">
                    {isVideo && item.poster_url ? (
                      <img src={item.poster_url} alt="poster" className="size-full object-cover transition-transform group-hover:scale-105" />
                    ) : isImage && mediaUrl(item) ? (
                      <img src={mediaUrl(item)} alt="image result" className="size-full object-cover transition-transform group-hover:scale-105" />
                    ) : !isVideo && item.art?.url ? (
                      <img src={item.art.url} alt="cover" className="size-full object-cover transition-transform group-hover:scale-105" />
                    ) : (
                      <div className="flex size-full items-center justify-center text-muted-foreground/60">
                        {isVideo ? <Video className="size-5" /> : isImage ? <ImageIcon className="size-5" /> : <Music2 className="size-5" />}
                      </div>
                    )}
                  </div>
                  <div className="min-w-0 flex-1 space-y-1">
                    <p className="truncate font-display text-sm font-semibold">{itemTitle(item)}</p>
                    <p className="truncate text-xs text-muted-foreground">{itemArtist(item)}</p>
                    <div className="flex flex-wrap gap-1">
                      <Badge variant={item.source === "result" ? "outline" : "muted"} className="text-[10px]">{sourceLabel(item)}</Badge>
                      {item.recommended && <Badge variant="default" className="text-[10px]">Recommended</Badge>}
                      {item.use_lora && <Badge variant="outline" className="text-[10px]">LoRA {item.lora_scale ?? ""}</Badge>}
                      {linkedVideos.length > 0 && (
                        <Badge variant="outline" className="gap-1 text-[10px]">
                          <Video className="size-3" /> {linkedVideos.length}
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2 pt-1 text-[10px] text-muted-foreground">
                      {item.duration && <span>{formatDuration(item.duration)}</span>}
                      {(item.song_model || item.model_label) && <span>· {item.song_model || item.model_label}</span>}
                    </div>
                  </div>
                </button>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </motion.div>

      <Dialog open={!!active} onOpenChange={(open) => !open && setActive(null)}>
        <DialogContent className="max-h-[90vh] max-w-3xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{active ? itemTitle(active) : "Library item"}</DialogTitle>
            <DialogDescription>
              {active ? `${sourceLabel(active)} · ${itemArtist(active)}` : ""}
            </DialogDescription>
          </DialogHeader>
          {active?.kind === "audio" && active.audio_url && <ActiveAudio active={active} />}
          {active?.kind === "image" && mediaUrl(active) && (
            <img
              src={mediaUrl(active)}
              alt={itemTitle(active)}
              className="max-h-[70vh] w-full rounded-lg border bg-black object-contain"
            />
          )}
          {active?.kind === "video" && mediaUrl(active) && (
            <video
              src={mediaUrl(active)}
              poster={active.poster_url || undefined}
              controls
              className="aspect-video w-full rounded-lg bg-black object-contain"
            />
          )}
          {active && (
            <div className="space-y-3">
              {active.kind === "audio" && (
                <MfluxArtMaker
                  title={itemTitle(active)}
                  artist={itemArtist(active)}
                  context={String(active.caption || active.tags || "")}
                  targetType={active.source === "song" ? "song" : "generation_result"}
                  targetId={active.source === "song" ? active.song_id : active.result_id}
                  compact
                />
              )}
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <Stat label="Source" value={sourceLabel(active)} />
                <Stat label="BPM" value={active.bpm} />
                <Stat label="Key" value={active.key_scale} />
                <Stat label="Duur" value={active.duration ? formatDuration(active.duration) : undefined} />
                <Stat label="Size" value={active.width && active.height ? `${active.width}×${active.height}` : undefined} />
              </div>
              <div className="flex flex-wrap gap-1">
                {tags(active).slice(0, 16).map((t) => (
                  <Badge key={t} variant="muted" className="text-[10px]">{t}</Badge>
                ))}
              </div>
              {active.kind === "audio" && activeVideos.length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Linked videos</p>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {activeVideos.map((video) => (
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
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div className="flex flex-wrap gap-2">
                {mediaUrl(active) && (
                  <Button asChild variant="outline" size="sm" className="gap-2">
                    <a href={mediaUrl(active)} download target="_blank" rel="noreferrer">
                      <Download className="size-3.5" /> Download
                    </a>
                  </Button>
                )}
                {active.kind === "audio" && active.audio_url && (
                  <Button asChild variant="outline" size="sm" className="gap-2">
                    <Link
                      to="/wizard/video"
                      state={{
                        audio_url: active.audio_url,
                        title: itemTitle(active),
                        artist_name: itemArtist(active),
                        prompt: String(active.caption || active.tags || itemTitle(active) || ""),
                        target_type: active.source === "song" ? "song" : "result",
                        target_id: active.song_id || active.result_id || active.id,
                      }}
                    >
                      <Video className="size-3.5" /> Create video
                    </Link>
                  </Button>
                )}
                {active.deletable && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setDeleteTarget(active);
                      setDeleteText("");
                    }}
                    className="gap-2 text-destructive hover:text-destructive"
                  >
                    <Trash2 className="size-3.5" /> Delete from disk
                  </Button>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      <Dialog open={!!deleteTarget} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Permanent verwijderen?</DialogTitle>
            <DialogDescription>
              Dit verwijdert het bestand echt van schijf. Typ DELETE om te bevestigen.
            </DialogDescription>
          </DialogHeader>
          {deleteTarget && (
            <div className="space-y-3">
              <div className="rounded-md border bg-card/40 p-3 text-sm">
                <div className="flex items-center gap-2">
                  <HardDrive className="size-4 text-destructive" />
                  <span className="font-medium">{itemTitle(deleteTarget)}</span>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{sourceLabel(deleteTarget)} · {mediaUrl(deleteTarget)}</p>
              </div>
              <Input value={deleteText} onChange={(e) => setDeleteText(e.target.value)} placeholder="DELETE" />
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteTarget(null)}>Annuleer</Button>
            <Button
              variant="destructive"
              disabled={deleteText !== "DELETE" || !deleteTarget || remove.isPending}
              onClick={() => deleteTarget && remove.mutate(deleteTarget)}
            >
              <Trash2 className={cn("size-4", remove.isPending && "animate-pulse")} />
              Delete from disk
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
