import * as React from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Wand2, ImagePlus, RefreshCw, Settings as SettingsIcon } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  generateArt,
  getLLMCatalog,
  imageModelDetails,
  type ArtMetadata,
  type ArtGenerateRequest,
  type LLMProvider,
} from "@/lib/api";
import { useSettingsStore } from "@/store/settings";
import { toast } from "@/components/ui/sonner";
import { cn } from "@/lib/utils";

interface ArtGeneratorProps {
  scope: "single" | "album";
  defaultPrompt: string;
  attachToResultId?: string;
  attachToAlbumFamilyId?: string;
  title?: string;
  caption?: string;
  className?: string;
  onCreated?: (art: ArtMetadata) => void;
}

export function ArtGenerator({
  scope,
  defaultPrompt,
  attachToResultId,
  attachToAlbumFamilyId,
  title,
  caption,
  className,
  onCreated,
}: ArtGeneratorProps) {
  const [prompt, setPrompt] = React.useState(defaultPrompt);
  const [art, setArt] = React.useState<ArtMetadata | null>(null);
  const artProvider = useSettingsStore((s) => s.artProvider);
  const artModel = useSettingsStore((s) => s.artModel);
  const setArt_ = useSettingsStore((s) => s.setArt);

  const catalogQuery = useQuery({
    queryKey: ["llm-catalog"],
    queryFn: getLLMCatalog,
    staleTime: 30_000,
  });

  const imageModels = React.useMemo(
    () => imageModelDetails(catalogQuery.data),
    [catalogQuery.data],
  );

  // Auto-pick: catalog.settings.art_model if installed
  React.useEffect(() => {
    if (artModel || !catalogQuery.data) return;
    const settings = catalogQuery.data.settings;
    if (
      settings?.art_model &&
      imageModels.some((m) => m.name === settings.art_model)
    ) {
      setArt_(
        (settings.art_provider ?? "ollama") as LLMProvider,
        settings.art_model,
      );
      return;
    }
    if (imageModels.length > 0) {
      setArt_(imageModels[0].provider, imageModels[0].name);
    }
  }, [catalogQuery.data, imageModels, artModel, setArt_]);

  React.useEffect(() => {
    if (defaultPrompt && !prompt) setPrompt(defaultPrompt);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultPrompt]);

  const mutation = useMutation({
    mutationFn: (override: Partial<ArtGenerateRequest> = {}) =>
      generateArt({
        scope,
        prompt,
        title,
        caption,
        model: artModel || undefined,
        attach_to_result_id: attachToResultId,
        attach_to_album_family_id: attachToAlbumFamilyId,
        ...override,
      }),
    onSuccess: (resp) => {
      if (!resp.success || !resp.art) {
        toast.error(resp.error || "Art generation mislukte");
        return;
      }
      setArt(resp.art);
      onCreated?.(resp.art);
      toast.success("Artwork gegenereerd.");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const isLoading = catalogQuery.isLoading;
  const isEmpty = !isLoading && imageModels.length === 0;

  return (
    <div className={cn("space-y-3 rounded-xl border bg-card/40 p-4", className)}>
      <div className="flex items-center justify-between gap-2">
        <h3 className="font-display text-sm font-semibold">
          {scope === "album" ? "Album cover" : "Track artwork"}
        </h3>
        {art && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => mutation.mutate({})}
            disabled={mutation.isPending}
            className="gap-1.5 text-xs"
          >
            <RefreshCw className="size-3" />
            Nieuwe variant
          </Button>
        )}
      </div>

      <div className="grid gap-3 sm:grid-cols-[160px_1fr]">
        <div className="aspect-square overflow-hidden rounded-lg border bg-background/40">
          {mutation.isPending ? (
            <Skeleton className="size-full" />
          ) : art?.url ? (
            <motion.img
              key={art.art_id}
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3 }}
              src={art.url}
              alt={title || "artwork"}
              className="size-full object-cover"
              loading="lazy"
            />
          ) : (
            <div className="flex size-full flex-col items-center justify-center gap-1 text-muted-foreground/60">
              <ImagePlus className="size-7" />
              <span className="text-[10px] uppercase tracking-widest">
                nog geen art
              </span>
            </div>
          )}
        </div>

        <div className="space-y-2">
          <div className="space-y-1">
            <Label className="text-xs uppercase tracking-wider text-muted-foreground">
              Prompt
            </Label>
            <Textarea
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs uppercase tracking-wider text-muted-foreground">
              Image-model
            </Label>
            {isEmpty ? (
              <div className="rounded-md border border-yellow-500/30 bg-yellow-500/5 p-2.5 text-xs text-yellow-200">
                Geen Ollama image-model geïnstalleerd. Voorbeeld:{" "}
                <code className="rounded bg-background/40 px-1">
                  ollama pull aravhawk/flux:11.9bf16
                </code>
                <Link
                  to="/settings"
                  className="ml-2 inline-flex items-center gap-1 underline"
                >
                  <SettingsIcon className="size-3" /> Open Settings
                </Link>
              </div>
            ) : (
              <Select
                value={artModel}
                onValueChange={(v) => {
                  const found = imageModels.find((m) => m.name === v);
                  if (found) setArt_(found.provider, found.name);
                }}
              >
                <SelectTrigger>
                  <SelectValue placeholder={isLoading ? "Modellen laden…" : "Kies een image-model"} />
                </SelectTrigger>
                <SelectContent>
                  {imageModels.map((m) => (
                    <SelectItem key={m.key} value={m.name}>
                      {m.display_name || m.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
          <Button
            onClick={() => mutation.mutate({})}
            disabled={!prompt.trim() || !artModel || mutation.isPending}
            className="w-full gap-2"
          >
            <Wand2 className="size-4" />
            {mutation.isPending ? "Genereert…" : art ? "Regenereer" : "Genereer artwork"}
          </Button>
        </div>
      </div>
    </div>
  );
}
