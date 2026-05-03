import * as React from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  Sparkles, ImagePlus, Download, CheckCircle2, XCircle, Loader2, Cpu, Disc, Brush,
  Layers, RefreshCw, Trash2, FlaskConical, Plus,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  api,
  chatModelDetails,
  embeddingModelDetails,
  getLLMCatalog,
  imageModelDetails,
  PROVIDER_LABEL,
  type LLMModelDetail,
  type LLMProvider,
} from "@/lib/api";
import { useSettingsStore } from "@/store/settings";
import { useJobsStore } from "@/store/jobs";
import { toast } from "@/components/ui/sonner";

function modelKey(provider: LLMProvider, name: string) {
  return `${provider}:${name}`;
}

// ---- LLM tab ------------------------------------------------------------

function LLMTab() {
  const qc = useQueryClient();
  const plannerProvider = useSettingsStore((s) => s.plannerProvider);
  const plannerModel = useSettingsStore((s) => s.plannerModel);
  const setPlanner = useSettingsStore((s) => s.setPlanner);
  const artProvider = useSettingsStore((s) => s.artProvider);
  const artModel = useSettingsStore((s) => s.artModel);
  const setArt = useSettingsStore((s) => s.setArt);
  const addJob = useJobsStore((s) => s.addJob);
  const updateJob = useJobsStore((s) => s.updateJob);
  const removeJob = useJobsStore((s) => s.removeJob);

  const catalogQuery = useQuery({
    queryKey: ["llm-catalog"],
    queryFn: getLLMCatalog,
    staleTime: 30_000,
  });

  const catalog = catalogQuery.data;
  const chatModels = chatModelDetails(catalog);
  const imageModels = imageModelDetails(catalog);
  const embeddingModels = embeddingModelDetails(catalog);

  // Group chat models by provider
  const groupedChat = React.useMemo(() => {
    const m = new Map<LLMProvider, LLMModelDetail[]>();
    for (const d of chatModels) {
      const arr = m.get(d.provider) ?? [];
      arr.push(d);
      m.set(d.provider, arr);
    }
    return m;
  }, [chatModels]);

  const testChat = useMutation({
    mutationFn: () =>
      api.post<{ success: boolean; reply?: string; error?: string }>(
        "/api/ollama/test",
        { model: plannerModel, prompt: "Say hi in five words." },
      ),
    onSuccess: (r) =>
      r.success
        ? toast.success(`Reply: ${r.reply ?? ""}`)
        : toast.error(r.error ?? "test mislukt"),
    onError: (e: Error) => toast.error(e.message),
  });

  // Ollama pull flow
  const [pullModelName, setPullModelName] = React.useState("");
  const startPull = useMutation({
    mutationFn: () =>
      api.post<{ success: boolean; job?: { id: string }; error?: string }>(
        "/api/ollama/pull",
        { model: pullModelName.trim(), reason: "manual settings" },
      ),
    onSuccess: (resp) => {
      if (!resp.success || !resp.job?.id) {
        toast.error(resp.error || "Pull starten mislukt");
        return;
      }
      const id = resp.job.id;
      addJob({
        id,
        kind: "ollama-pull",
        label: `pull ${pullModelName.trim()}`,
        progress: 0,
        status: "queued",
        startedAt: Date.now(),
      });
      setPullModelName("");
      pollPull(id);
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const pollPull = (jobId: string) => {
    const tick = async () => {
      try {
        const resp = await api.get<{
          success: boolean;
          job?: { status?: string; progress?: number; error?: string };
        }>(`/api/ollama/pull/${encodeURIComponent(jobId)}`);
        const j = resp.job;
        if (!j) return;
        updateJob(jobId, { progress: j.progress, status: j.status });
        if (j.status === "complete") {
          toast.success("Ollama-model klaar.");
          setTimeout(() => removeJob(jobId), 4000);
          qc.invalidateQueries({ queryKey: ["llm-catalog"] });
          return;
        }
        if (j.status === "error") {
          toast.error(j.error || "Pull mislukt");
          setTimeout(() => removeJob(jobId), 6000);
          return;
        }
        setTimeout(tick, 2500);
      } catch (e) {
        toast.error(`Poll fout: ${(e as Error).message}`);
      }
    };
    tick();
  };

  const providerCard = (id: LLMProvider) => {
    const sub = catalog?.catalogs?.[id];
    const provider = catalog?.providers.find((p) => p.id === id);
    return (
      <Card key={id}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="size-4" /> {PROVIDER_LABEL[id]}
            {provider && (
              provider.ready
                ? <Badge className="gap-1"><CheckCircle2 className="size-3" /> ready</Badge>
                : <Badge variant="destructive" className="gap-1"><XCircle className="size-3" /> offline</Badge>
            )}
          </CardTitle>
          <CardDescription>
            {provider?.host ?? sub?.host ?? "—"} · {sub?.details.length ?? 0} model{(sub?.details.length ?? 0) === 1 ? "" : "len"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-1.5">
          {!sub || sub.details.length === 0 ? (
            <p className="text-xs text-muted-foreground">Geen modellen geïnstalleerd.</p>
          ) : (
            sub.details.map((m) => (
              <div
                key={m.key}
                className="flex items-center gap-2 rounded-md border bg-card/40 px-2 py-1.5 text-xs"
              >
                <span className="flex-1 truncate font-mono">
                  {m.profile?.dropdown_label || m.display_name || m.name}
                </span>
                {m.size_gb && (
                  <span className="text-[10px] text-muted-foreground">
                    {m.size_gb.toFixed(1)} GB
                  </span>
                )}
                {m.kind === "embedding" && (
                  <Badge variant="muted" className="text-[10px]">embedding</Badge>
                )}
                {m.image_generation && (
                  <Badge variant="muted" className="text-[10px]">image</Badge>
                )}
                {m.kind === "chat" && (
                  <Badge variant="muted" className="text-[10px]">chat</Badge>
                )}
              </div>
            ))
          )}
          {sub?.error && <p className="text-xs text-destructive">{sub.error}</p>}
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-4">
      {/* Provider overview cards */}
      <div className="grid gap-3 lg:grid-cols-3">
        {providerCard("ollama")}
        {providerCard("lmstudio")}
        {providerCard("ace_step_lm")}
      </div>

      {/* Planner selection */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="size-4 text-primary" /> AI planner (chat)
          </CardTitle>
          <CardDescription>
            Vult elke wizard automatisch in via /api/prompt-assistant/run.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Label>Model</Label>
          {catalogQuery.isLoading ? (
            <Skeleton className="h-9 w-full" />
          ) : chatModels.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Nog geen chat-modellen. Pull er eentje hieronder.
            </p>
          ) : (
            <Select
              value={plannerModel ? modelKey(plannerProvider, plannerModel) : ""}
              onValueChange={(key) => {
                const idx = key.indexOf(":");
                if (idx > 0) {
                  setPlanner(key.slice(0, idx) as LLMProvider, key.slice(idx + 1));
                }
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Kies een planner-model" />
              </SelectTrigger>
              <SelectContent>
                {Array.from(groupedChat.entries()).map(([provider, list]) => (
                  <SelectGroup key={provider}>
                    <SelectLabel>{PROVIDER_LABEL[provider]}</SelectLabel>
                    {list.map((m) => (
                      <SelectItem key={m.key} value={modelKey(m.provider, m.name)}>
                        {m.profile?.dropdown_label || m.display_name || m.name}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                ))}
              </SelectContent>
            </Select>
          )}
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="outline"
              disabled={!plannerModel || testChat.isPending}
              onClick={() => testChat.mutate()}
              className="gap-1.5"
            >
              <FlaskConical className="size-3" /> Test
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => qc.invalidateQueries({ queryKey: ["llm-catalog"] })}
              className="gap-1.5"
            >
              <RefreshCw className="size-3" /> Refresh
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Art model */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ImagePlus className="size-4 text-primary" /> Album- &amp; track-art
          </CardTitle>
          <CardDescription>
            Ollama image-model dat artwork rendert.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <Label>Model</Label>
          {catalogQuery.isLoading ? (
            <Skeleton className="h-9 w-full" />
          ) : imageModels.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              Geen image-modellen. Pull er eentje, bijvoorbeeld:
              <code className="mx-1 rounded bg-background/40 px-1">aravhawk/flux:11.9bf16</code>
            </p>
          ) : (
            <Select
              value={artModel}
              onValueChange={(v) => {
                const found = imageModels.find((m) => m.name === v);
                if (found) setArt(found.provider, found.name);
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Kies een image-model" />
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
        </CardContent>
      </Card>

      {/* Pull a new Ollama model */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Download className="size-4" /> Pull Ollama-model
          </CardTitle>
          <CardDescription>
            Voer een Ollama tag in (bv. <code className="rounded bg-background/40 px-1">qwen3:14b</code>)
            en download op de achtergrond. Je ziet de voortgang in de sidebar.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex gap-2">
          <Input
            placeholder="model:tag"
            value={pullModelName}
            onChange={(e) => setPullModelName(e.target.value)}
            className="flex-1"
          />
          <Button
            onClick={() => startPull.mutate()}
            disabled={!pullModelName.trim() || startPull.isPending}
            className="gap-1.5"
          >
            <Plus className="size-3" />
            Start pull
          </Button>
        </CardContent>
      </Card>

      {/* Embeddings (info-only) */}
      {embeddingModels.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Embedding-modellen</CardTitle>
            <CardDescription>Gebruikt door album-coherence.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-1.5">
            {embeddingModels.map((m) => (
              <div key={m.key} className="rounded-md border bg-card/30 p-2 text-xs">
                <span className="font-mono">{m.name}</span>
                <span className="ml-2 text-[10px] text-muted-foreground">
                  {PROVIDER_LABEL[m.provider]}
                </span>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ---- Models tab ---------------------------------------------------------

function ModelsTab() {
  const downloadsQ = useQuery({
    queryKey: ["models", "downloads"],
    queryFn: () =>
      api.get<{
        success: boolean;
        models?: Array<{ name: string; installed?: boolean; size?: number; description?: string; recommended?: boolean }>;
      }>("/api/models/downloads"),
  });
  const startDownload = useMutation({
    mutationFn: (model: string) =>
      api.post<{ success: boolean; job_id?: string; error?: string }>("/api/models/download", { model }),
    onSuccess: (r) => r.success ? toast.success("Download gestart.") : toast.error(r.error ?? "download fout"),
    onError: (e: Error) => toast.error(e.message),
  });
  const models = downloadsQ.data?.models ?? [];
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Disc className="size-4" /> Song-modellen
        </CardTitle>
        <CardDescription>ACE-Step v1.5 base/sft/turbo varianten.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {downloadsQ.isLoading ? (
          Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-12 rounded-md" />)
        ) : models.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            Geen modellen-lijst beschikbaar.
          </p>
        ) : (
          models.map((m) => (
            <div
              key={m.name}
              className="flex items-center gap-3 rounded-md border bg-card/40 p-3"
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className="truncate font-mono text-sm">{m.name}</p>
                  {m.recommended && <Badge variant="outline" className="text-[10px]">aanbevolen</Badge>}
                  {m.installed && <Badge className="text-[10px]">geïnstalleerd</Badge>}
                </div>
                {m.description && <p className="truncate text-xs text-muted-foreground">{m.description}</p>}
              </div>
              {m.installed ? (
                <Button variant="ghost" size="sm" disabled className="gap-1.5">
                  <CheckCircle2 className="size-3.5" />
                  Aanwezig
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => startDownload.mutate(m.name)}
                  disabled={startDownload.isPending}
                  className="gap-1.5"
                >
                  <Download className="size-3.5" /> Download
                </Button>
              )}
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}

// ---- LoRA tab -----------------------------------------------------------

function LoRATab() {
  const qc = useQueryClient();
  const status = useQuery({
    queryKey: ["lora", "status"],
    queryFn: () =>
      api.get<{
        success: boolean;
        loaded?: { name: string; scale: number; path?: string };
        available?: Array<{ name: string; path: string }>;
      }>("/api/lora/status"),
  });

  const [scale, setScale] = React.useState<number>(1);
  React.useEffect(() => {
    if (status.data?.loaded) setScale(status.data.loaded.scale ?? 1);
  }, [status.data?.loaded]);

  const loadLora = useMutation({
    mutationFn: (path: string) => api.post<{ success: boolean; error?: string }>("/api/lora/load", { path }),
    onSuccess: (r) => {
      if (!r.success) toast.error(r.error ?? "load fout");
      else { toast.success("LoRA geladen."); qc.invalidateQueries({ queryKey: ["lora"] }); }
    },
    onError: (e: Error) => toast.error(e.message),
  });
  const unloadLora = useMutation({
    mutationFn: () => api.post<{ success: boolean }>("/api/lora/unload"),
    onSuccess: () => { toast.success("LoRA ontladen."); qc.invalidateQueries({ queryKey: ["lora"] }); },
    onError: (e: Error) => toast.error(e.message),
  });
  const setScaleMutation = useMutation({
    mutationFn: (s: number) => api.post<{ success: boolean }>("/api/lora/scale", { scale: s }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["lora"] }),
    onError: (e: Error) => toast.error(e.message),
  });

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="size-4" /> Geladen LoRA
          </CardTitle>
          <CardDescription>Een LoRA die nu actief is, beïnvloedt elke generatie.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {status.isLoading ? (
            <Skeleton className="h-12 w-full" />
          ) : status.data?.loaded ? (
            <>
              <div className="flex items-center gap-3 rounded-md border bg-card/40 p-3">
                <Brush className="size-4 text-primary" />
                <div className="flex-1 min-w-0">
                  <p className="truncate font-medium">{status.data.loaded.name}</p>
                  {status.data.loaded.path && (
                    <p className="truncate text-[10px] text-muted-foreground">{status.data.loaded.path}</p>
                  )}
                </div>
                <Button size="sm" variant="ghost" onClick={() => unloadLora.mutate()} className="gap-1.5 text-destructive">
                  <Trash2 className="size-3.5" /> Unload
                </Button>
              </div>
              <div className="space-y-2">
                <div className="flex items-baseline justify-between">
                  <Label>Scale</Label>
                  <span className="font-mono text-xs">{scale.toFixed(2)}</span>
                </div>
                <Slider
                  value={[scale]}
                  min={0}
                  max={1.5}
                  step={0.01}
                  onValueChange={(v) => setScale(v[0] ?? 1)}
                  onValueCommit={(v) => setScaleMutation.mutate(v[0] ?? 1)}
                />
              </div>
            </>
          ) : (
            <p className="text-sm text-muted-foreground">Geen LoRA geladen.</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Beschikbare LoRA's</CardTitle>
          <CardDescription>Eerder getrainde of geïmporteerde adapters.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {status.isLoading ? (
            <Skeleton className="h-10 w-full" />
          ) : (status.data?.available ?? []).length === 0 ? (
            <p className="text-sm text-muted-foreground">Nog geen LoRA's. Train er één in de Trainer wizard.</p>
          ) : (
            (status.data?.available ?? []).map((l) => (
              <div key={l.path} className="flex items-center gap-2 rounded-md border bg-card/30 p-2">
                <span className="flex-1 truncate text-sm">{l.name}</span>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => loadLora.mutate(l.path)}
                  disabled={loadLora.isPending}
                >
                  Load
                </Button>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// ---- Settings page ------------------------------------------------------

export function Settings() {
  return (
    <div className="mx-auto w-full max-w-4xl space-y-6 px-6 py-10 sm:px-10">
      <header className="space-y-1">
        <h1 className="font-display text-3xl font-semibold">Settings</h1>
        <p className="text-sm text-muted-foreground">
          Configureer je AI-providers, modellen en LoRA's.
        </p>
      </header>

      <Tabs defaultValue="llm" className="w-full">
        <TabsList>
          <TabsTrigger value="llm">AI &amp; Art</TabsTrigger>
          <TabsTrigger value="models">Modellen</TabsTrigger>
          <TabsTrigger value="lora">LoRA</TabsTrigger>
        </TabsList>
        <TabsContent value="llm" className="mt-4">
          <LLMTab />
        </TabsContent>
        <TabsContent value="models" className="mt-4">
          <ModelsTab />
        </TabsContent>
        <TabsContent value="lora" className="mt-4">
          <LoRATab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
