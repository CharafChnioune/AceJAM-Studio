import * as React from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  Sparkles, Download, CheckCircle2, XCircle, Loader2, Cpu, Disc, Brush, Image as ImageIcon,
  Layers, RefreshCw, Trash2, FlaskConical, Plus, Video,
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
  getMfluxLoras,
  getMfluxModels,
  getMfluxStatus,
  getMlxVideoLoras,
  getMlxVideoModels,
  getMlxVideoStatus,
  registerMlxVideoModelDir,
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
  const embeddingProvider = useSettingsStore((s) => s.embeddingProvider);
  const embeddingModel = useSettingsStore((s) => s.embeddingModel);
  const setEmbedding = useSettingsStore((s) => s.setEmbedding);
  const addJob = useJobsStore((s) => s.addJob);
  const updateJob = useJobsStore((s) => s.updateJob);

  const catalogQuery = useQuery({
    queryKey: ["llm-catalog"],
    queryFn: getLLMCatalog,
    staleTime: 30_000,
  });

  const catalog = catalogQuery.data;
  const chatModels = chatModelDetails(catalog);
  const embeddingModels = React.useMemo(() => {
    const priority = (name: string) => {
      const lower = name.toLowerCase();
      if (lower === "charaf/qwen3-vl-embedding-8b:latest") return 0;
      if (lower === "mxbai-embed-large:latest") return 1;
      if (lower === "nomic-embed-text:latest") return 2;
      return 3;
    };
    return [...embeddingModelDetails(catalog)].sort((a, b) => {
      const pa = priority(a.name);
      const pb = priority(b.name);
      if (pa !== pb) return pa - pb;
      return a.name.localeCompare(b.name);
    });
  }, [catalog]);

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

  const groupedEmbedding = React.useMemo(() => {
    const m = new Map<LLMProvider, LLMModelDetail[]>();
    for (const d of embeddingModels) {
      const arr = m.get(d.provider) ?? [];
      arr.push(d);
      m.set(d.provider, arr);
    }
    return m;
  }, [embeddingModels]);

  const saveLocalSettings = useMutation({
    mutationFn: (patch: Record<string, unknown>) =>
      api.post<{ success: boolean; settings?: Record<string, unknown>; error?: string }>(
        "/api/local-llm/settings",
        {
          ...(catalog?.settings ?? {}),
          provider: plannerProvider,
          chat_model: plannerModel,
          embedding_provider: embeddingProvider,
          embedding_model: embeddingModel,
          ...patch,
        },
      ),
    onSuccess: (r) => {
      if (!r.success) {
        toast.error(r.error || "Settings bewaren mislukt");
        return;
      }
      qc.invalidateQueries({ queryKey: ["llm-catalog"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });

  React.useEffect(() => {
    if (embeddingModel || !catalog) return;
    const preferred = String(catalog.settings?.embedding_model || "");
    const preferredProvider = (catalog.settings?.embedding_provider ?? "ollama") as LLMProvider;
    if (preferred && embeddingModels.some((m) => m.provider === preferredProvider && m.name === preferred)) {
      setEmbedding(preferredProvider, preferred);
      return;
    }
    const first = embeddingModels[0];
    if (first) {
      setEmbedding(first.provider, first.name);
    }
  }, [catalog, embeddingModel, embeddingModels, setEmbedding]);

  const testChat = useMutation({
    mutationFn: () =>
      api.post<{ success: boolean; response?: string; reply?: string; error?: string }>(
        "/api/local-llm/test",
        { provider: plannerProvider, model: plannerModel, kind: "chat", prompt: "Say hi in five words." },
      ),
    onSuccess: (r) =>
      r.success
        ? toast.success(`Reply: ${r.reply ?? r.response ?? ""}`)
        : toast.error(r.error ?? "test mislukt"),
    onError: (e: Error) => toast.error(e.message),
  });

  const testEmbedding = useMutation({
    mutationFn: () =>
      api.post<{ success: boolean; dimensions?: number; error?: string }>(
        "/api/local-llm/test",
        { provider: embeddingProvider, model: embeddingModel, kind: "embedding" },
      ),
    onSuccess: (r) =>
      r.success
        ? toast.success(`Embedding OK${r.dimensions ? ` (${r.dimensions} dims)` : ""}`)
        : toast.error(r.error ?? "embedding-test mislukt"),
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
        state: "queued",
        kindLabel: "Ollama pull",
        detailsPath: `/api/ollama/pull/${encodeURIComponent(id)}`,
        metadata: {
          model: pullModelName.trim(),
          reason: "manual settings",
        },
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
        updateJob(jobId, {
          progress: j.progress,
          status: j.status,
          state: j.status,
          detailsPath: `/api/ollama/pull/${encodeURIComponent(jobId)}`,
          metadata: j as Record<string, unknown>,
          error: j.error,
        });
        if (j.status === "complete") {
          toast.success("Ollama-model klaar.");
          updateJob(jobId, { status: "complete", state: "complete", progress: 100 });
          qc.invalidateQueries({ queryKey: ["llm-catalog"] });
          return;
        }
        if (j.status === "error") {
          toast.error(j.error || "Pull mislukt");
          updateJob(jobId, { status: "error", state: "error", error: j.error });
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
                  const provider = key.slice(0, idx) as LLMProvider;
                  const model = key.slice(idx + 1);
                  setPlanner(provider, model);
                  saveLocalSettings.mutate({ provider, chat_model: model });
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

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="size-4 text-primary" /> AI Memory / RAG embeddings
          </CardTitle>
          <CardDescription>
            Gebruikt door CrewAI album-memory, RAG en knowledge retrieval. ACE-Step audio text encoder blijft: Qwen3-Embedding-0.6B.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Label>Embedding model</Label>
          {catalogQuery.isLoading ? (
            <Skeleton className="h-9 w-full" />
          ) : embeddingModels.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Geen embedding-modellen gevonden. Pull er eentje hieronder, bijvoorbeeld{" "}
              <code className="rounded bg-background/40 px-1">mxbai-embed-large:latest</code>
              {" "}of selecteer een embeddingmodel in LM Studio.
            </p>
          ) : (
            <Select
              value={embeddingModel ? modelKey(embeddingProvider, embeddingModel) : ""}
              onValueChange={(key) => {
                const idx = key.indexOf(":");
                if (idx > 0) {
                  const provider = key.slice(0, idx) as LLMProvider;
                  const model = key.slice(idx + 1);
                  setEmbedding(provider, model);
                  saveLocalSettings.mutate({
                    embedding_provider: provider,
                    embedding_model: model,
                  });
                }
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Kies een embedding-model" />
              </SelectTrigger>
              <SelectContent>
                {Array.from(groupedEmbedding.entries()).map(([provider, list]) => (
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
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              disabled={!embeddingModel || testEmbedding.isPending}
              onClick={() => testEmbedding.mutate()}
              className="gap-1.5"
            >
              <FlaskConical className="size-3" /> Test embedding
            </Button>
            {embeddingModel && (
              <Badge variant="muted" className="font-mono text-[10px]">
                {PROVIDER_LABEL[embeddingProvider]} · {embeddingModel}
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>
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

// ---- MFLUX tab ----------------------------------------------------------

function MFLUXTab() {
  const status = useQuery({ queryKey: ["mflux", "status"], queryFn: getMfluxStatus, staleTime: 30_000 });
  const models = useQuery({ queryKey: ["mflux", "models"], queryFn: getMfluxModels, staleTime: 60_000 });
  const loras = useQuery({ queryKey: ["mflux", "loras"], queryFn: getMfluxLoras, staleTime: 20_000 });
  const statusData = status.data;
  const grouped = React.useMemo(() => {
    const map = new Map<string, NonNullable<typeof models.data>["models"]>();
    for (const model of models.data?.models ?? []) {
      const arr = map.get(model.preset) ?? [];
      arr.push(model);
      map.set(model.preset, arr);
    }
    return Array.from(map.entries());
  }, [models.data?.models]);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ImageIcon className="size-4 text-primary" /> MFLUX runtime
          </CardTitle>
          <CardDescription>
            MLX Media gebruikt MFLUX voor image generation, edits, upscale, depth en image-LoRAs.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {status.isLoading ? (
            <Skeleton className="h-16 rounded-md lg:col-span-4" />
          ) : (
            <>
              <div className="rounded-md border bg-card/40 p-3 text-xs">
                <p className="text-muted-foreground">Platform</p>
                <p className="mt-1 font-mono">{statusData?.platform} / {statusData?.arch}</p>
              </div>
              <div className="rounded-md border bg-card/40 p-3 text-xs">
                <p className="text-muted-foreground">Apple Silicon</p>
                <Badge className="mt-1" variant={statusData?.apple_silicon ? "default" : "destructive"}>
                  {statusData?.apple_silicon ? "ready" : "blocked"}
                </Badge>
              </div>
              <div className="rounded-md border bg-card/40 p-3 text-xs">
                <p className="text-muted-foreground">MLX</p>
                <Badge className="mt-1" variant={statusData?.mlx_available ? "default" : "destructive"}>
                  {statusData?.mlx_available ? "installed" : "missing"}
                </Badge>
              </div>
              <div className="rounded-md border bg-card/40 p-3 text-xs">
                <p className="text-muted-foreground">MFLUX</p>
                <Badge className="mt-1" variant={statusData?.mflux_available || statusData?.cli_available ? "default" : "destructive"}>
                  {statusData?.mflux_available || statusData?.cli_available ? "installed" : "missing"}
                </Badge>
              </div>
              {statusData?.blocking_reason && (
                <p className="rounded-md bg-destructive/10 p-3 text-xs text-destructive sm:col-span-2 lg:col-span-4">
                  {statusData.blocking_reason}
                </p>
              )}
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>MFLUX commands</CardTitle>
          <CardDescription>Health checks for the console scripts used by each Image Studio action.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-2 md:grid-cols-2">
            {Object.entries(statusData?.action_readiness ?? {}).map(([action, info]) => (
              <div key={action} className="rounded-md border bg-card/35 p-3 text-sm">
                <div className="flex items-center justify-between gap-2">
                  <p className="font-medium">{info.label || action}</p>
                  <Badge variant={info.ready ? "default" : "destructive"}>{info.ready ? "ready" : "missing"}</Badge>
                </div>
                <p className="mt-1 font-mono text-[10px] text-muted-foreground">
                  {(info.available_commands?.length ? info.available_commands : info.commands || []).join(", ") || "no command"}
                </p>
                {!info.ready && info.reason && <p className="mt-1 text-xs text-muted-foreground">{info.reason}</p>}
              </div>
            ))}
          </div>
          <div className="grid gap-2 md:grid-cols-3">
            {[
              ["Data", statusData?.data_dir],
              ["Uploads", statusData?.uploads_dir],
              ["LoRAs", statusData?.lora_dir],
            ].map(([label, value]) => (
              <div key={String(label)} className="rounded-md border bg-background/35 p-3">
                <p className="text-xs text-muted-foreground">{label}</p>
                <p className="mt-1 truncate font-mono text-[10px]">{value || "default"}</p>
              </div>
            ))}
          </div>
          <div className="max-h-40 overflow-auto rounded-md border bg-background/35 p-3">
            {Object.entries(statusData?.commands ?? {}).map(([command, path]) => (
              <div key={command} className="flex items-center justify-between gap-3 py-1 text-xs">
                <span className="font-mono">{command}</span>
                <span className={path ? "truncate text-muted-foreground" : "text-destructive"}>{path || "missing"}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Model presets</CardTitle>
          <CardDescription>Max Quality, LoRA/Training, Fast Draft, Edit/Upscale/Depth.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {models.isLoading ? (
            <Skeleton className="h-32 rounded-md" />
          ) : (
            grouped.map(([preset, list]) => (
              <div key={preset} className="space-y-2">
                <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{preset}</p>
                <div className="grid gap-2 md:grid-cols-2">
                  {list.map((model) => (
                    <div key={model.id} className="rounded-md border bg-card/35 p-3">
                      <div className="flex items-center justify-between gap-2">
                        <p className="font-medium">{model.label}</p>
                        {model.trainable && <Badge variant="outline" className="text-[10px]">trainable</Badge>}
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground">{model.description}</p>
                      <p className="mt-2 font-mono text-[10px] text-muted-foreground">
                        {model.command || model.id} · q{model.quantization_default ?? 8} · {model.default_steps ?? "auto"} steps
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Image-LoRA registry</CardTitle>
          <CardDescription>Adapters in <code>app/data/mflux/loras</code>.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {loras.isLoading ? (
            <Skeleton className="h-12 rounded-md" />
          ) : (loras.data?.adapters ?? []).length === 0 ? (
            <p className="text-sm text-muted-foreground">Nog geen image-LoRAs geregistreerd.</p>
          ) : (
            (loras.data?.adapters ?? []).map((adapter) => (
              <div key={adapter.path} className="rounded-md border bg-card/35 p-3">
                <div className="flex items-center gap-2">
                  <p className="truncate font-medium">{adapter.display_name || adapter.trigger_tag || adapter.name}</p>
                  <Badge variant="outline" className="text-[10px]">{adapter.model_id || "MFLUX"}</Badge>
                </div>
                <p className="mt-1 truncate font-mono text-[10px] text-muted-foreground">{adapter.path}</p>
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// ---- MLX Video tab ------------------------------------------------------

function MLXVideoTab() {
  const qc = useQueryClient();
  const [modelDirPath, setModelDirPath] = React.useState("");
  const [modelDirLabel, setModelDirLabel] = React.useState("");
  const [modelDirFamily, setModelDirFamily] = React.useState("wan21");
  const status = useQuery({ queryKey: ["mlx-video", "status"], queryFn: getMlxVideoStatus, staleTime: 30_000 });
  const models = useQuery({ queryKey: ["mlx-video", "models"], queryFn: getMlxVideoModels, staleTime: 60_000 });
  const loras = useQuery({ queryKey: ["mlx-video", "loras"], queryFn: getMlxVideoLoras, staleTime: 20_000 });
  const statusData = status.data;
  const grouped = React.useMemo(() => {
    const map = new Map<string, NonNullable<typeof models.data>["models"]>();
    for (const model of models.data?.models ?? []) {
      const arr = map.get(model.preset || "other") ?? [];
      arr.push(model);
      map.set(model.preset || "other", arr);
    }
    return Array.from(map.entries());
  }, [models.data?.models]);

  const registerModelDir = useMutation({
    mutationFn: () => registerMlxVideoModelDir({ path: modelDirPath.trim(), label: modelDirLabel.trim(), family: modelDirFamily }),
    onSuccess: (resp) => {
      if (!resp.success) {
        toast.error(resp.error || "Model-dir registreren mislukt");
        return;
      }
      toast.success("Wan model-dir geregistreerd.");
      setModelDirPath("");
      setModelDirLabel("");
      qc.invalidateQueries({ queryKey: ["mlx-video"] });
    },
    onError: (error: Error) => toast.error(error.message),
  });

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Video className="size-4 text-primary" /> MLX video runtime
          </CardTitle>
          <CardDescription>
            Video Studio gebruikt een aparte <code>app/video-env</code> met Python 3.11 zodat audio en images stabiel blijven.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {status.isLoading ? (
            <Skeleton className="h-16 rounded-md lg:col-span-4" />
          ) : (
            <>
              <div className="rounded-md border bg-card/40 p-3 text-xs">
                <p className="text-muted-foreground">Platform</p>
                <p className="mt-1 font-mono">{statusData?.platform} / {statusData?.arch}</p>
              </div>
              <div className="rounded-md border bg-card/40 p-3 text-xs">
                <p className="text-muted-foreground">Python</p>
                <Badge className="mt-1" variant={statusData?.python?.ok ? "default" : "destructive"}>
                  {statusData?.python?.version || "missing"}
                </Badge>
              </div>
              <div className="rounded-md border bg-card/40 p-3 text-xs">
                <p className="text-muted-foreground">MLX</p>
                <Badge className="mt-1" variant={statusData?.mlx_available ? "default" : "destructive"}>
                  {statusData?.mlx_available ? "installed" : "missing"}
                </Badge>
              </div>
              <div className="rounded-md border bg-card/40 p-3 text-xs">
                <p className="text-muted-foreground">mlx-video</p>
                <Badge className="mt-1" variant={statusData?.mlx_video_available ? "default" : "destructive"}>
                  {statusData?.mlx_video_available ? "installed" : "missing"}
                </Badge>
              </div>
              {statusData?.blocking_reason && (
                <p className="rounded-md bg-destructive/10 p-3 text-xs text-destructive sm:col-span-2 lg:col-span-4">
                  {statusData.blocking_reason}
                </p>
              )}
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Commands & upstream fixes</CardTitle>
          <CardDescription>LTX/Wan command discovery, PR patch status en issue #26 guard.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-2 md:grid-cols-2">
            {Object.entries(statusData?.command_help ?? {}).map(([engine, info]) => (
              <div key={engine} className="rounded-md border bg-card/35 p-3 text-sm">
                <div className="flex items-center justify-between gap-2">
                  <p className="font-medium">{engine.toUpperCase()}</p>
                  <Badge variant={info.help_ok ? "default" : "destructive"}>{info.help_ok ? "ready" : "missing"}</Badge>
                </div>
                <p className="mt-1 truncate font-mono text-[10px] text-muted-foreground">
                  {(info.command || statusData?.commands?.[engine] || []).join(" ")}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">Output: {info.output_flag || "--output-path"}</p>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {Object.entries(info.capabilities ?? {}).filter(([, value]) => Boolean(value)).slice(0, 9).map(([name]) => (
                    <Badge key={name} variant="outline" className="text-[10px]">{name.replaceAll("_", " ")}</Badge>
                  ))}
                </div>
                {!info.help_ok && info.reason && <p className="mt-1 text-xs text-muted-foreground">{info.reason}</p>}
              </div>
            ))}
          </div>
          <div className="grid gap-2 md:grid-cols-4">
            {[
              ["LTX-2.3 VAE fix", statusData?.patch_status?.vae_fix_active],
              ["PR #23 end frame", statusData?.patch_status?.pr23_ltx_i2v_end_frame],
              ["Issue #26 guard", statusData?.patch_status?.tokenizer_issue_26_guarded],
              ["Helios PR #21", statusData?.patch_status?.helios_pr21_enabled],
            ].map(([label, value]) => (
              <div key={String(label)} className="rounded-md border bg-background/35 p-3">
                <p className="text-xs text-muted-foreground">{String(label)}</p>
                <Badge className="mt-1" variant={value ? "default" : "outline"}>{value ? "on" : "off"}</Badge>
              </div>
            ))}
          </div>
          <div className="rounded-md border bg-background/35 p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-sm font-medium">Video-LoRA training</p>
                <p className="text-xs text-muted-foreground">
                  {String(statusData?.patch_status?.training_reason || "Upstream exposes no stable train command yet.")}
                </p>
              </div>
              <Badge variant={statusData?.patch_status?.training_available ? "default" : "outline"}>
                {statusData?.patch_status?.training_available ? "available" : "not available upstream"}
              </Badge>
            </div>
          </div>
          <div className="grid gap-2 md:grid-cols-3">
            {[
              ["Video env", statusData?.video_env_dir],
              ["Results", statusData?.results_dir],
              ["LoRAs", statusData?.lora_dir],
            ].map(([label, value]) => (
              <div key={String(label)} className="rounded-md border bg-background/35 p-3">
                <p className="text-xs text-muted-foreground">{label}</p>
                <p className="mt-1 truncate font-mono text-[10px]">{String(value || "default")}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Video presets</CardTitle>
          <CardDescription>Draft-first: snelle LTX preview, Wan 480P wanneer je model-dir registreert, daarna Final/HQ.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {models.isLoading ? (
            <Skeleton className="h-32 rounded-md" />
          ) : (
            grouped.map(([preset, list]) => (
              <div key={preset} className="space-y-2">
                <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">{preset}</p>
                <div className="grid gap-2 md:grid-cols-2">
                  {list.map((model) => (
                    <div key={model.id} className="rounded-md border bg-card/35 p-3">
                      <div className="flex items-center justify-between gap-2">
                        <p className="font-medium">{model.label}</p>
                        <Badge variant={model.disabled ? "destructive" : "outline"} className="text-[10px]">{model.engine}</Badge>
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground">{model.description}</p>
                      <p className="mt-2 font-mono text-[10px] text-muted-foreground">
                        {model.default_width}×{model.default_height} · {model.default_frames} frames · {model.default_steps ?? "auto"} steps
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Wan model directories</CardTitle>
          <CardDescription>Wan-modellen worden niet stil gedownload. Registreer hier je geconverteerde MLX model folders.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-2 md:grid-cols-[1fr_180px_140px_auto]">
            <Input value={modelDirPath} onChange={(event) => setModelDirPath(event.target.value)} placeholder="/Volumes/SSD/Wan2.1-T2V-1.3B-MLX" />
            <Input value={modelDirLabel} onChange={(event) => setModelDirLabel(event.target.value)} placeholder="Label" />
            <Select value={modelDirFamily} onValueChange={setModelDirFamily}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="wan21">Wan2.1</SelectItem>
                <SelectItem value="wan22">Wan2.2</SelectItem>
                <SelectItem value="wan">Wan auto</SelectItem>
              </SelectContent>
            </Select>
            <Button onClick={() => registerModelDir.mutate()} disabled={!modelDirPath.trim() || registerModelDir.isPending}>
              {registerModelDir.isPending ? <Loader2 className="size-4 animate-spin" /> : <Plus className="size-4" />}
              Add
            </Button>
          </div>
          {(statusData?.registered_model_dirs ?? []).length === 0 ? (
            <p className="text-sm text-muted-foreground">Nog geen Wan model-dir geregistreerd.</p>
          ) : (
            <div className="space-y-2">
              {(statusData?.registered_model_dirs ?? []).map((dir) => (
                <div key={dir.path} className="rounded-md border bg-card/35 p-3">
                  <div className="flex items-center gap-2">
                    <p className="truncate font-medium">{dir.label}</p>
                    <Badge variant="outline" className="text-[10px]">{dir.family || "wan"}</Badge>
                  </div>
                  <p className="mt-1 truncate font-mono text-[10px] text-muted-foreground">{dir.path}</p>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Video-LoRA registry</CardTitle>
          <CardDescription>Adapters in <code>app/data/mlx_video/loras</code>, inclusief Wan high/low rollen.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {loras.isLoading ? (
            <Skeleton className="h-12 rounded-md" />
          ) : (loras.data?.adapters ?? []).length === 0 ? (
            <p className="text-sm text-muted-foreground">Nog geen video-LoRAs geregistreerd.</p>
          ) : (
            (loras.data?.adapters ?? []).map((adapter) => (
              <div key={adapter.path} className="rounded-md border bg-card/35 p-3">
                <div className="flex items-center gap-2">
                  <p className="truncate font-medium">{adapter.display_name || adapter.name}</p>
                  <Badge variant="outline" className="text-[10px]">{adapter.family || "video"}</Badge>
                  <Badge variant="muted" className="text-[10px]">{adapter.role || "shared"}</Badge>
                </div>
                <p className="mt-1 truncate font-mono text-[10px] text-muted-foreground">{adapter.path}</p>
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
          <TabsTrigger value="llm">AI</TabsTrigger>
          <TabsTrigger value="mflux">MFLUX</TabsTrigger>
          <TabsTrigger value="video">Video</TabsTrigger>
          <TabsTrigger value="models">Modellen</TabsTrigger>
          <TabsTrigger value="lora">LoRA</TabsTrigger>
        </TabsList>
        <TabsContent value="llm" className="mt-4">
          <LLMTab />
        </TabsContent>
        <TabsContent value="mflux" className="mt-4">
          <MFLUXTab />
        </TabsContent>
        <TabsContent value="video" className="mt-4">
          <MLXVideoTab />
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
