import * as React from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { AlertTriangle, Clock3, Loader2, Sparkles, Wand2 } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  api,
  chatModelDetails,
  embeddingModelDetails,
  getAlbumPlanJob,
  getLLMCatalog,
  promptAssistantRun,
  PROVIDER_LABEL,
  startAlbumPlanJob,
  type LLMProvider,
  type WizardMode,
} from "@/lib/api";
import { useSettingsStore } from "@/store/settings";
import { normalizePasteBlocks, normalizeWarnings, useWizardStore } from "@/store/wizard";
import { toast } from "@/components/ui/sonner";

interface AIPromptStepProps {
  mode: WizardMode;
  placeholder: string;
  examples?: string[];
  currentPayload?: Record<string, unknown>;
  onHydrated?: (payload: Record<string, unknown>) => void;
  onPendingChange?: (pending: boolean) => void;
}

function modelKey(provider: LLMProvider, name: string): string {
  return `${provider}:${name}`;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function asText(value: unknown): string {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}

function asNumber(value: unknown, fallback = 0): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function AIPromptStep({
  mode,
  placeholder,
  examples,
  currentPayload,
  onHydrated,
  onPendingChange,
}: AIPromptStepProps) {
  const prompt = useWizardStore((s) => s.prompts[mode]) ?? "";
  const warnings = normalizeWarnings(useWizardStore((s) => s.warnings[mode]));
  const pasteBlocks = normalizePasteBlocks(useWizardStore((s) => s.pasteBlocks[mode]));
  const setPrompt = useWizardStore((s) => s.setPrompt);
  const setHydration = useWizardStore((s) => s.setHydration);

  const plannerProvider = useSettingsStore((s) => s.plannerProvider);
  const plannerModel = useSettingsStore((s) => s.plannerModel);
  const setPlanner = useSettingsStore((s) => s.setPlanner);
  const embeddingProvider = useSettingsStore((s) => s.embeddingProvider);
  const embeddingModel = useSettingsStore((s) => s.embeddingModel);
  const setEmbedding = useSettingsStore((s) => s.setEmbedding);

  const catalogQuery = useQuery({
    queryKey: ["llm-catalog"],
    queryFn: getLLMCatalog,
    staleTime: 30_000,
  });

  const allChatModels = React.useMemo(
    () => chatModelDetails(catalogQuery.data),
    [catalogQuery.data],
  );

  const allEmbeddingModels = React.useMemo(
    () =>
      embeddingModelDetails(catalogQuery.data).filter(
        (m) => m.provider === "ollama" || m.provider === "lmstudio",
      ),
    [catalogQuery.data],
  );

  // Group by provider for the dropdowns
  const groupedChatModels = React.useMemo(() => {
    const map = new Map<LLMProvider, typeof allChatModels>();
    for (const m of allChatModels) {
      const arr = map.get(m.provider) ?? [];
      arr.push(m);
      map.set(m.provider, arr);
    }
    return map;
  }, [allChatModels]);

  const groupedEmbeddingModels = React.useMemo(() => {
    const map = new Map<LLMProvider, typeof allEmbeddingModels>();
    for (const m of allEmbeddingModels) {
      const arr = map.get(m.provider) ?? [];
      arr.push(m);
      map.set(m.provider, arr);
    }
    return map;
  }, [allEmbeddingModels]);

  // Auto-pick a sensible default planner from catalog.settings or first chat model
  React.useEffect(() => {
    if (plannerModel || !catalogQuery.data) return;
    const settings = catalogQuery.data.settings;
    const preferred = settings?.chat_model;
    const preferredProvider = (settings?.provider ?? "ollama") as LLMProvider;
    if (preferred && allChatModels.some((m) => m.provider === preferredProvider && m.name === preferred)) {
      setPlanner(preferredProvider, preferred);
      return;
    }
    if (allChatModels.length > 0) {
      const first = allChatModels[0];
      setPlanner(first.provider, first.name);
    }
  }, [catalogQuery.data, allChatModels, plannerModel, setPlanner]);

  React.useEffect(() => {
    if (embeddingModel || !catalogQuery.data) return;
    const settings = catalogQuery.data.settings;
    const preferred = settings?.embedding_model;
    const preferredProvider = (settings?.embedding_provider ?? "ollama") as LLMProvider;
    if (
      preferred &&
      allEmbeddingModels.some((m) => m.provider === preferredProvider && m.name === preferred)
    ) {
      setEmbedding(preferredProvider, preferred);
      return;
    }
    if (allEmbeddingModels.length > 0) {
      const first = allEmbeddingModels[0];
      setEmbedding(first.provider, first.name);
    }
  }, [allEmbeddingModels, catalogQuery.data, embeddingModel, setEmbedding]);

  const currentKey = plannerModel ? modelKey(plannerProvider, plannerModel) : "";
  const embeddingKey = embeddingModel ? modelKey(embeddingProvider, embeddingModel) : "";
  const [albumJobId, setAlbumJobId] = React.useState("");
  const [albumJob, setAlbumJob] = React.useState<Record<string, unknown> | null>(null);

  const saveLocalSettings = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      api.post<{ success: boolean; error?: string }>("/api/local-llm/settings", body),
    onSuccess: (data) => {
      if (!data.success) toast.error(data.error || "AI Memory-instelling opslaan mislukt");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const aiFill = useMutation({
    mutationFn: () => {
      const payload = currentPayload ?? {};
      if (mode === "album") {
        return startAlbumPlanJob({
          ...payload,
          concept: prompt.trim() || asText(payload.concept),
          user_prompt: prompt,
          prompt,
          agent_engine: asText(payload.agent_engine) || "crewai_micro",
          album_writer_mode: "per_track_writer_loop",
          planner_lm_provider: plannerProvider,
          planner_model: plannerModel || undefined,
          ollama_model: plannerProvider === "ollama" ? plannerModel || undefined : asText(payload.ollama_model) || undefined,
          embedding_provider: embeddingProvider,
          embedding_lm_provider: embeddingProvider,
          embedding_model: embeddingModel || catalogQuery.data?.settings?.embedding_model || undefined,
        });
      }
      return promptAssistantRun({
        mode,
        user_prompt: prompt,
        current_payload: currentPayload,
        planner_lm_provider: plannerProvider,
        planner_model: plannerModel || undefined,
        embedding_provider: embeddingProvider,
        embedding_lm_provider: embeddingProvider,
        embedding_model: embeddingModel || catalogQuery.data?.settings?.embedding_model || undefined,
      });
    },
    onMutate: () => {
      if (mode === "album") {
        const totalTracks = asNumber((currentPayload ?? {}).num_tracks, 0);
        setAlbumJobId("");
        setAlbumJob({
          state: "starting",
          stage: "starting",
          status: "Album AI job starten",
          current_task: "Backend job-id ophalen",
          current_agent: "Album planner",
          current_track: 0,
          total_tracks: totalTracks || "?",
          completed_tracks: 0,
          remaining_tracks: totalTracks || "?",
          progress: 1,
          waiting_on_llm: false,
          logs: ["Album AI job wordt gestart via de background planner."],
        });
      }
    },
    onSuccess: (data) => {
      const response = data as {
        success?: boolean;
        error?: string;
        message?: string;
        logs?: string[];
        job_id?: string;
        job?: unknown;
        ollama_pull_started?: boolean;
        ollama_model?: string;
        ollama_pull_job?: unknown;
      };
      if (mode === "album" && response.ollama_pull_started) {
        const pullJob = asRecord(response.ollama_pull_job);
        const model = asText(response.ollama_model) || asText(pullJob.model) || "Ollama model";
        const progress = asNumber(pullJob.progress, 0);
        const status = asText(pullJob.status) || "pulling model";
        const logs = Array.isArray(response.logs)
          ? response.logs.map(String)
          : Array.isArray(pullJob.logs)
            ? pullJob.logs.map(String)
            : [asText(response.message) || `${model} wordt gedownload.`];
        setAlbumJobId("");
        setAlbumJob({
          id: asText(pullJob.id),
          state: asText(pullJob.state) || "pulling",
          stage: "model_download",
          status,
          current_task: `Planner-model downloaden: ${model}`,
          current_agent: "Local LLM install",
          progress,
          waiting_on_llm: false,
          logs,
        });
        toast.info(`${model} wordt eerst gedownload. Start Album AI Fill opnieuw zodra de download klaar is.`);
        return;
      }
      if (!response.success) {
        toast.error(response.error || "AI-fill mislukte");
        return;
      }
      if (mode === "album" && "job_id" in data) {
        const jobId = asText(response.job_id);
        if (!jobId) {
          toast.error("Album AI job kon niet worden gestart.");
          return;
        }
        setAlbumJobId(jobId);
        setAlbumJob(asRecord(response.job));
        toast.success("Album AI Fill gestart. Je ziet live welke taak bezig is.");
        return;
      }
      const assistantData = data as {
        payload?: Record<string, unknown>;
        warnings?: string[] | string | null;
        paste_blocks?: Array<{ label?: string; content?: string; text?: string }> | string | null;
      };
      setHydration(mode, {
        payload: assistantData.payload,
        warnings: assistantData.warnings,
        paste_blocks: assistantData.paste_blocks,
      });
      if (assistantData.payload) onHydrated?.(assistantData.payload);
      toast.success("AI heeft het wizard-formulier voorgevuld.");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  React.useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      if (!albumJobId) return;
      try {
        const data = await getAlbumPlanJob(albumJobId);
        if (cancelled) return;
        const job = asRecord(data.job);
        setAlbumJob(job);
        const state = asText(job.state).toLowerCase();
        if (state === "succeeded") {
          const result = asRecord(job.result);
          const payload = asRecord(result.payload).tracks ? asRecord(result.payload) : result;
          setHydration(mode, {
            payload,
            warnings: result.warnings,
            paste_blocks: result.paste_blocks,
          });
          if (Object.keys(payload).length > 0) onHydrated?.(payload);
          setAlbumJobId("");
          toast.success("Album AI heeft het wizard-formulier voorgevuld.");
          return;
        }
        if (state === "failed" || state === "needs_review") {
          const errors = Array.isArray(job.errors) ? job.errors.map(String) : [];
          setAlbumJobId("");
          toast.error(errors[0] || asText(job.error) || "Album AI Fill mislukte");
          return;
        }
      } catch (err) {
        if (!cancelled) toast.error(err instanceof Error ? err.message : "Album jobstatus laden mislukt");
      }
      if (!cancelled) timer = setTimeout(poll, 2000);
    };
    if (albumJobId) {
      timer = setTimeout(poll, 600);
    }
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [albumJobId, mode, onHydrated, setHydration]);

  const albumFillPending = Boolean(albumJobId);
  const pending = aiFill.isPending || albumFillPending;

  React.useEffect(() => {
    // Parity anchor: onPendingChange?.(aiFill.isPending) is now extended with album job polling.
    onPendingChange?.(pending);
  }, [pending, onPendingChange]);

  const onModelChange = (key: string) => {
    if (!key) return;
    const idx = key.indexOf(":");
    if (idx < 0) return;
    const provider = key.slice(0, idx) as LLMProvider;
    const name = key.slice(idx + 1);
    setPlanner(provider, name);
  };

  const onEmbeddingChange = (key: string) => {
    if (!key) return;
    const idx = key.indexOf(":");
    if (idx < 0) return;
    const provider = key.slice(0, idx) as LLMProvider;
    const name = key.slice(idx + 1);
    setEmbedding(provider, name);
    saveLocalSettings.mutate({
      embedding_provider: provider,
      embedding_model: name,
    });
  };

  const isLoading = catalogQuery.isLoading;
  const isEmpty = !isLoading && allChatModels.length === 0;
  const embeddingsEmpty = !isLoading && allEmbeddingModels.length === 0;
  const albumWaiting =
    Boolean(albumJob?.waiting_on_llm) ||
    asText(albumJob?.waiting_on_llm).toLowerCase() === "true";

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <Label htmlFor={`${mode}-prompt`} className="text-sm font-medium">
          Wat wil je maken?
        </Label>
        <Textarea
          id={`${mode}-prompt`}
          placeholder={placeholder}
          value={prompt}
          onChange={(e) => setPrompt(mode, e.target.value)}
          rows={6}
          className="text-base leading-relaxed"
        />
        {examples && examples.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            <span className="text-xs text-muted-foreground">Voorbeelden:</span>
            {examples.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => setPrompt(mode, ex)}
                className="rounded-full border border-border/60 px-2.5 py-0.5 text-xs text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground"
              >
                {ex}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] lg:items-end">
        <div className="space-y-2">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">
            Planner model
          </Label>
          {isEmpty ? (
            <div className="rounded-md border border-yellow-500/30 bg-yellow-500/5 p-3 text-xs text-yellow-200">
              Geen chat-modellen gevonden bij Ollama, LM Studio of de officiële
              ACE-Step LM. Open <Link to="/settings" className="underline">Settings</Link> of
              install een Ollama model met <code className="rounded bg-background/40 px-1">ollama pull qwen3:14b</code>.
            </div>
          ) : (
            <Select value={currentKey} onValueChange={onModelChange}>
              <SelectTrigger>
                <SelectValue placeholder={isLoading ? "Catalog laden…" : "Kies een planner-model"} />
              </SelectTrigger>
              <SelectContent>
                {Array.from(groupedChatModels.entries()).map(([provider, list]) => (
                  <SelectGroup key={provider}>
                    <SelectLabel>{PROVIDER_LABEL[provider]}</SelectLabel>
                    {list.map((m) => {
                      const dropdownLabel =
                        m.profile?.dropdown_label || m.display_name || m.name;
                      return (
                        <SelectItem key={m.key} value={modelKey(m.provider, m.name)}>
                          <div className="flex items-center gap-2">
                            <span>{dropdownLabel}</span>
                            {m.size_gb && (
                              <span className="text-[10px] text-muted-foreground">
                                {m.size_gb.toFixed(1)} GB
                              </span>
                            )}
                          </div>
                        </SelectItem>
                      );
                    })}
                  </SelectGroup>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
        <div className="space-y-2">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">
            AI Memory / RAG embedding
          </Label>
          {embeddingsEmpty ? (
            <div className="rounded-md border border-yellow-500/30 bg-yellow-500/5 p-3 text-xs text-yellow-200">
              Geen embedding-modellen gevonden bij Ollama of LM Studio. Open{" "}
              <Link to="/settings" className="underline">Settings</Link>{" "}
              en kies of pull een embedding-model voor CrewAI memory/RAG.
            </div>
          ) : (
            <Select value={embeddingKey} onValueChange={onEmbeddingChange}>
              <SelectTrigger>
                <SelectValue placeholder={isLoading ? "Catalog laden…" : "Kies embedding-model"} />
              </SelectTrigger>
              <SelectContent>
                {Array.from(groupedEmbeddingModels.entries()).map(([provider, list]) => (
                  <SelectGroup key={provider}>
                    <SelectLabel>{PROVIDER_LABEL[provider]}</SelectLabel>
                    {list.map((m) => {
                      const dropdownLabel =
                        m.profile?.dropdown_label || m.display_name || m.name;
                      return (
                        <SelectItem key={m.key} value={modelKey(m.provider, m.name)}>
                          <div className="flex items-center gap-2">
                            <span>{dropdownLabel}</span>
                            {m.size_gb && (
                              <span className="text-[10px] text-muted-foreground">
                                {m.size_gb.toFixed(1)} GB
                              </span>
                            )}
                          </div>
                        </SelectItem>
                      );
                    })}
                  </SelectGroup>
                ))}
              </SelectContent>
            </Select>
          )}
          <p className="text-[11px] text-muted-foreground">
            Alleen voor AI/CrewAI memory en RAG. ACE-Step audio text encoder blijft Qwen3-Embedding-0.6B.
          </p>
        </div>
        <Button
          size="lg"
          onClick={() => aiFill.mutate()}
          disabled={!prompt.trim() || !plannerModel || pending}
          className="gap-2"
        >
          {pending ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
          {pending ? (mode === "album" ? "Album AI loopt…" : "AI denkt na…") : "Vul met AI"}
        </Button>
      </div>

      {mode === "album" && albumJob && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-3 rounded-xl border border-primary/25 bg-primary/5 p-4"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <div className="text-sm font-semibold">Album AI is bezig</div>
              <div className="text-xs text-muted-foreground">
                {asText(albumJob.current_task) || asText(albumJob.status) || "Taak wordt voorbereid"}
              </div>
            </div>
            <Badge variant={albumWaiting ? "secondary" : "outline"}>
              {asText(albumJob.stage) || asText(albumJob.state) || "running"}
            </Badge>
          </div>
          <Progress value={Math.max(0, Math.min(100, asNumber(albumJob.progress, 0)))} />
          <div className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-3">
            <div>
              Track{" "}
              <span className="font-medium text-foreground">
                {asText(albumJob.current_track) || "0"}/{asText(albumJob.total_tracks) || "?"}
              </span>
            </div>
            <div>
              Klaar{" "}
              <span className="font-medium text-foreground">
                {asText(albumJob.completed_tracks) || "0"}
              </span>{" "}
              · Nog{" "}
              <span className="font-medium text-foreground">
                {asText(albumJob.remaining_tracks) || "?"}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Clock3 className="size-3" />
              {albumWaiting
                ? `Wacht op ${asText(albumJob.llm_provider) || "LLM"} ${asText(albumJob.llm_wait_elapsed_s) || "0"}s`
                : asText(albumJob.current_agent) || "CrewAI"}
            </div>
          </div>
          {Array.isArray(albumJob.logs) && albumJob.logs.length > 0 && (
            <div className="rounded-md border border-border/50 bg-background/50 p-2 text-xs text-muted-foreground">
              {String(albumJob.logs[albumJob.logs.length - 1])}
            </div>
          )}
        </motion.div>
      )}

      {(warnings.length > 0 || pasteBlocks.length > 0) && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-3 rounded-xl border border-border/60 bg-card/40 p-4"
        >
          {warnings.length > 0 && (
            <div className="flex gap-2 text-sm">
              <AlertTriangle className="size-4 shrink-0 text-yellow-400" />
              <ul className="space-y-1">
                {warnings.map((w, i) => (
                  <li key={i} className="text-yellow-200/90">{w}</li>
                ))}
              </ul>
            </div>
          )}
          {pasteBlocks.length > 0 && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">
                <Wand2 className="size-3" /> Paste blocks
              </div>
              {pasteBlocks.map((b, i) => (
                <details key={i} className="rounded-md border border-border/40 bg-background/40 p-2 text-xs">
                  <summary className="cursor-pointer font-medium">
                    {b.label || `Block ${i + 1}`}
                  </summary>
                  <pre className="mt-2 whitespace-pre-wrap font-mono text-[11px] text-muted-foreground">
                    {b.content}
                  </pre>
                </details>
              ))}
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
