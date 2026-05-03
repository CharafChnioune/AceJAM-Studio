import * as React from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  Sparkles, ImagePlus, Download, CheckCircle2, XCircle, Loader2, Cpu, Disc, Brush,
  Layers, RefreshCw, Trash2, FlaskConical,
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
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { api, listLocalLLMModels } from "@/lib/api";
import { useSettingsStore } from "@/store/settings";
import { toast } from "@/components/ui/sonner";
import { cn } from "@/lib/utils";

// ---- LLM tab ------------------------------------------------------------

function LLMTab() {
  const planner = useSettingsStore((s) => s.plannerModel);
  const setPlanner = useSettingsStore((s) => s.setPlanner);
  const artModel = useSettingsStore((s) => s.artModel);
  const setArtModel = useSettingsStore((s) => s.setArtModel);

  const chatQ = useQuery({
    queryKey: ["local-llm-models", "chat"],
    queryFn: () => listLocalLLMModels("chat"),
  });
  const imageQ = useQuery({
    queryKey: ["local-llm-models", "image_generation"],
    queryFn: () => listLocalLLMModels("image_generation"),
  });
  const ollamaStatus = useQuery({
    queryKey: ["ollama-status"],
    queryFn: () =>
      api.get<{
        success: boolean;
        ready: boolean;
        ollama_host: string;
        model_count: number;
        chat_model_count: number;
        embedding_model_count: number;
        running_models?: Array<{ name?: string }>;
        error?: string;
      }>("/api/ollama/status"),
  });

  const testChat = useMutation({
    mutationFn: () =>
      api.post<{ success: boolean; reply?: string; error?: string }>(
        "/api/ollama/test",
        { model: planner, prompt: "Say hi in five words." },
      ),
    onSuccess: (r) => r.success ? toast.success(`Reply: ${r.reply ?? ""}`) : toast.error(r.error ?? "test mislukt"),
    onError: (e: Error) => toast.error(e.message),
  });

  const status = ollamaStatus.data;
  const ready = status?.ready;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="size-4" /> Ollama-status
          </CardTitle>
          <CardDescription>Bron van zowel planner-LLM als image-generator.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-3">
            {ollamaStatus.isLoading ? (
              <Skeleton className="h-6 w-32" />
            ) : (
              <>
                {ready ? (
                  <Badge className="gap-1"><CheckCircle2 className="size-3" /> Online</Badge>
                ) : (
                  <Badge variant="destructive" className="gap-1"><XCircle className="size-3" /> Offline</Badge>
                )}
                <code className="rounded bg-background/40 px-2 py-1 font-mono text-xs">
                  {status?.ollama_host}
                </code>
                <span className="text-xs text-muted-foreground">
                  {status?.chat_model_count ?? 0} chat · {status?.embedding_model_count ?? 0} embed · {status?.model_count ?? 0} totaal
                </span>
                <Button size="sm" variant="ghost" onClick={() => ollamaStatus.refetch()} className="ml-auto gap-1.5">
                  <RefreshCw className="size-3" /> Refresh
                </Button>
              </>
            )}
          </div>
          {status?.error && (
            <p className="mt-2 text-xs text-destructive">{status.error}</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="size-4 text-primary" /> AI planner (chat)
          </CardTitle>
          <CardDescription>Vult elke wizard automatisch in via /api/prompt-assistant/run.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Label>Model</Label>
          {chatQ.isLoading ? (
            <Skeleton className="h-9 w-full" />
          ) : (
            <Select value={planner} onValueChange={(v) => setPlanner("ollama", v)}>
              <SelectTrigger><SelectValue placeholder="Kies een planner-model" /></SelectTrigger>
              <SelectContent>
                {(chatQ.data?.models ?? []).map((m) => (
                  <SelectItem key={m.name} value={m.name}>
                    <div className="flex items-center gap-2">
                      <span>{m.name}</span>
                      {m.installed === false && <Badge variant="muted" className="text-[10px]">niet geïnstalleerd</Badge>}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          <div className="flex gap-2">
            <Button size="sm" variant="outline" disabled={!planner || testChat.isPending} onClick={() => testChat.mutate()} className="gap-1.5">
              <FlaskConical className="size-3" /> Test
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ImagePlus className="size-4 text-primary" /> Album- & track-art
          </CardTitle>
          <CardDescription>Ollama image-model dat artwork rendert.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <Label>Model</Label>
          {imageQ.isLoading ? (
            <Skeleton className="h-9 w-full" />
          ) : (
            <Select value={artModel} onValueChange={setArtModel}>
              <SelectTrigger><SelectValue placeholder="Kies een image-model" /></SelectTrigger>
              <SelectContent>
                {(imageQ.data?.models ?? []).map((m) => (
                  <SelectItem key={m.name} value={m.name}>{m.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          {(imageQ.data?.models ?? []).length === 0 && (
            <p className="text-xs text-muted-foreground">
              Geen image-modellen geïnstalleerd. Installeer er één met
              <code className="mx-1 rounded bg-background/40 px-1">ollama pull &lt;model&gt;</code>
              of via de Models-tab.
            </p>
          )}
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
