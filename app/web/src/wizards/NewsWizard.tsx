import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useForm, Controller } from "react-hook-form";
import { motion } from "framer-motion";
import { Newspaper, Music4, Hash, Video } from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { AIPromptStep } from "@/components/wizard/AIPromptStep";
import { ReviewStep } from "@/components/wizard/ReviewStep";
import { MusicQueueEditor } from "@/components/wizard/MusicQueueEditor";
import { GenerationJobStatus } from "@/components/wizard/GenerationJobStatus";
import { GenerationAudioList, firstGenerationAudioUrl } from "@/components/wizard/GenerationAudioList";
import { LoraSelector } from "@/components/wizard/LoraSelector";
import { RenderInsightPanel } from "@/components/wizard/RenderInsightPanel";
import { TagInput } from "@/components/wizard/TagInput";
import { AutomationFields } from "@/components/wizard/AutomationFields";
import { AudioStyleSelector } from "@/components/wizard/AudioStyleSelector";
import { AudioBackendSelector } from "@/components/wizard/AudioBackendSelector";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useGenerationJobRunner } from "@/hooks/useGenerationJobRunner";
import { mergeWizardDraft, usePromptMirror, useWizardDraft } from "@/hooks/useWizardDraft";
import { useWizardStore } from "@/store/wizard";
import { DEFAULT_LORA_SCALE, normalizeLoraSelection, type LoraSelection } from "@/lib/lora";
import { ACE_STEP_LANGUAGE_OPTIONS } from "@/lib/languages";
import { DEFAULT_AUDIO_BACKEND, audioBackendLabel, useMlxDitForAudioBackend } from "@/lib/audioBackend";
import { collectPayloadBlockingIssues, collectValidationMessages } from "@/lib/formValidation";
import { startSongBatchJob } from "@/lib/api";
import { mergePayloadWithCompanion, stripPromptCompanion, summarizeQueueEntry } from "@/lib/musicQueue";
import { formatDuration } from "@/lib/utils";
import { aceStepRenderDefaults } from "@/lib/aceStepSettings";

const MODE = "news" as const;

const SATIRE_MODES = [
  ["auto", "Auto (laat AI kiezen)"],
  ["funny_rap", "Funny rap"],
  ["pop_story", "Pop story"],
  ["club_banger", "Club banger"],
  ["drill_report", "Drill report"],
] as const;

interface NewsForm {
  user_prompt: string;
  title: string;
  artist_name: string;
  news_angle: string;
  satire_mode: string;
  caption: string;
  style_profile: string;
  tags: string;
  negative_tags: string;
  lyrics: string;
  duration: number;
  bpm?: number;
  key_scale?: string;
  vocal_language: string;
  song_model: string;
  audio_backend: "mlx";
  batch_size: number;
  use_lora: boolean;
  lora_adapter_path: string;
  lora_adapter_name: string;
  use_lora_trigger: boolean;
  lora_trigger_tag: string;
  lora_trigger_tags: string[];
  lora_scale: number;
  lora_adapters: LoraSelection["lora_adapters"];
  adapter_model_variant: string;
  adapter_song_model: string;
  payload_gate_status: string;
  payload_gate_passed: boolean;
  payload_gate_blocking_issues: string[];
  payload_quality_gate: Record<string, unknown>;
  rap_quality_report: Record<string, unknown>;
  rap_rewrite_status: string;
  rap_blocking_issues: string[];
  rap_strengths: string[];
  rap_revision_focus: string[];
  genre_execution_contract: Record<string, unknown>;
  lyric_technique_report: Record<string, unknown>;
  lora_selection_reason: string;
  performance_notes: string;
  strict_completion_notes: string;
  single_art_negative_prompt: string;
  video_negative_prompt: string;
  social_post_caption?: string;
  social_hook_line?: string;
  social_hashtags?: string;
  auto_song_art: boolean;
  auto_album_art: boolean;
  auto_video_clip: boolean;
  art_prompt: string;
  video_prompt: string;
}

export function NewsWizard() {
  const navigate = useNavigate();
  const setResult = useWizardStore((s) => s.setResult);
  const setPasteBlocks = useWizardStore((s) => s.setPasteBlocks);
  const lastResult = useWizardStore((s) => s.lastResult[MODE]);
  const warnings = useWizardStore((s) => s.warnings[MODE]) ?? [];
  const promptValidation = useWizardStore((s) => s.validations[MODE]);
  const storedQueue = useWizardStore((s) => s.queues[MODE]) ?? [];
  const setHydration = useWizardStore((s) => s.setHydration);
  const storePrompt = useWizardStore((s) => s.prompts[MODE]);
  const draft = useWizardStore((s) => s.drafts[MODE]);
  const newsDefaults = React.useMemo<NewsForm>(
    () => ({
      user_prompt: "",
      title: "",
      artist_name: "",
      news_angle: "",
      satire_mode: "auto",
      caption: "",
      style_profile: "auto",
      tags: "",
      negative_tags: "",
      lyrics: "",
      duration: 90,
      vocal_language: "nl",
      song_model: "acestep-v15-xl-sft",
      audio_backend: DEFAULT_AUDIO_BACKEND,
      batch_size: 1,
      use_lora: false,
      lora_adapter_path: "",
      lora_adapter_name: "",
      use_lora_trigger: false,
      lora_trigger_tag: "",
      lora_trigger_tags: [],
      lora_scale: DEFAULT_LORA_SCALE,
      lora_adapters: [],
      adapter_model_variant: "",
      adapter_song_model: "",
      payload_gate_status: "",
      payload_gate_passed: false,
      payload_gate_blocking_issues: [],
      payload_quality_gate: {},
      rap_quality_report: {},
      rap_rewrite_status: "",
      rap_blocking_issues: [],
      rap_strengths: [],
      rap_revision_focus: [],
      genre_execution_contract: {},
      lyric_technique_report: {},
      lora_selection_reason: "",
      performance_notes: "",
      strict_completion_notes: "",
      single_art_negative_prompt: "",
      video_negative_prompt: "",
      social_post_caption: "",
      social_hook_line: "",
      social_hashtags: "",
      auto_song_art: false,
      auto_album_art: false,
      auto_video_clip: false,
      art_prompt: "",
      video_prompt: "",
    }),
    [],
  );

  const form = useForm<NewsForm>({
    defaultValues: mergeWizardDraft<NewsForm>(newsDefaults, draft),
    mode: "onChange",
  });

  const [step, setStep] = React.useState(0);
  const [aiPromptPending, setAiPromptPending] = React.useState(false);
  const [queue, setQueue] = React.useState<Record<string, unknown>[]>(storedQueue);
  const [editingQueueIndex, setEditingQueueIndex] = React.useState(-1);
  const values = form.watch();
  const reviewBlockingIssues = React.useMemo(
    () => [
      ...collectValidationMessages(form.formState.errors),
      ...collectPayloadBlockingIssues(promptValidation, form.getValues()),
    ],
    [form, form.formState.errors, promptValidation],
  );
  const draftState = useWizardDraft(MODE, form);

  usePromptMirror(form, "user_prompt", storePrompt);

  const generation = useGenerationJobRunner({
    mode: MODE,
    label: "news track",
    onComplete: (resp) => {
      setResult(MODE, resp as unknown as Record<string, unknown>);
      setStep(999);
    },
  });

  const setLoraSelection = (selection: LoraSelection) => {
    form.setValue("use_lora", selection.use_lora, { shouldValidate: true });
    form.setValue("lora_adapter_path", selection.lora_adapter_path, { shouldValidate: true });
    form.setValue("lora_adapter_name", selection.lora_adapter_name, { shouldValidate: true });
    form.setValue("use_lora_trigger", selection.use_lora_trigger, { shouldValidate: true });
    form.setValue("lora_trigger_tag", selection.lora_trigger_tag, { shouldValidate: true });
    form.setValue("lora_trigger_tags", selection.lora_trigger_tags, { shouldValidate: true });
    form.setValue("lora_scale", selection.lora_scale, { shouldValidate: true });
    form.setValue("lora_adapters", selection.lora_adapters, { shouldValidate: true });
    form.setValue("adapter_model_variant", selection.adapter_model_variant, { shouldValidate: true });
    form.setValue("adapter_song_model", selection.adapter_song_model, { shouldValidate: true });
    if (selection.use_lora && selection.adapter_song_model) {
      form.setValue("song_model", selection.adapter_song_model, { shouldValidate: true });
    }
  };

  const buildPayload = () => {
    const v = form.getValues();
    const renderDefaults = aceStepRenderDefaults(v.song_model, "chart_master");
    return {
      task_type: "text2music",
      title: v.title,
      artist_name: v.artist_name,
      news_angle: v.news_angle,
      satire_mode: v.satire_mode,
      caption: v.caption,
      style_profile: v.style_profile,
      tags: v.tags,
      negative_tags: v.negative_tags,
      lyrics: v.lyrics,
      instrumental: false,
      audio_duration: v.duration,
      duration: v.duration,
      bpm: v.bpm,
      key_scale: v.key_scale,
      vocal_language: v.vocal_language || "nl",
      song_model: v.song_model,
      audio_backend: v.audio_backend,
      use_mlx_dit: useMlxDitForAudioBackend(v.audio_backend),
      quality_profile: renderDefaults.quality_profile,
      inference_steps: renderDefaults.inference_steps,
      guidance_scale: renderDefaults.guidance_scale,
      shift: renderDefaults.shift,
      audio_format: renderDefaults.audio_format,
      infer_method: renderDefaults.infer_method,
      sampler_mode: renderDefaults.sampler_mode,
      use_adg: renderDefaults.use_adg,
      batch_size: v.batch_size,
      variant_count: v.batch_size,
      payload_gate_status: v.payload_gate_status,
      payload_gate_passed: v.payload_gate_passed,
      payload_gate_blocking_issues: v.payload_gate_blocking_issues,
      payload_quality_gate: v.payload_quality_gate,
      rap_quality_report: v.rap_quality_report,
      rap_rewrite_status: v.rap_rewrite_status,
      rap_blocking_issues: v.rap_blocking_issues,
      rap_strengths: v.rap_strengths,
      rap_revision_focus: v.rap_revision_focus,
      auto_song_art: v.auto_song_art,
      auto_album_art: false,
      auto_video_clip: v.auto_video_clip,
      art_prompt: v.art_prompt,
      video_prompt: v.video_prompt,
      ...normalizeLoraSelection(v),
      social_pack: {
        post_caption: v.social_post_caption,
        hook_line: v.social_hook_line,
        hashtags: Array.isArray(v.social_hashtags)
          ? v.social_hashtags.map((item) => String(item ?? "").trim()).filter(Boolean)
          : String(v.social_hashtags || "").split(/[ ,]+/).filter(Boolean),
      },
    };
  };

  const buildCompanion = () => {
    const v = form.getValues();
    return {
      genre_execution_contract: v.genre_execution_contract,
      lyric_technique_report: v.lyric_technique_report,
      lora_selection_reason: v.lora_selection_reason,
      performance_notes: v.performance_notes,
      strict_completion_notes: v.strict_completion_notes,
      single_art_negative_prompt: v.single_art_negative_prompt,
      video_negative_prompt: v.video_negative_prompt,
      payload_gate_status: v.payload_gate_status,
      payload_gate_passed: v.payload_gate_passed,
      payload_gate_blocking_issues: v.payload_gate_blocking_issues,
      payload_quality_gate: v.payload_quality_gate,
      rap_quality_report: v.rap_quality_report,
      rap_rewrite_status: v.rap_rewrite_status,
      rap_blocking_issues: v.rap_blocking_issues,
      rap_strengths: v.rap_strengths,
      rap_revision_focus: v.rap_revision_focus,
    };
  };

  const buildDraftPayload = () => mergePayloadWithCompanion(buildPayload(), buildCompanion());

  const syncQueueState = React.useCallback(
    (nextQueue: Record<string, unknown>[]) => {
      setQueue(nextQueue);
      setHydration(MODE, {
        payload: buildPayload(),
        validation: promptValidation ?? null,
        warnings,
        companion: buildCompanion(),
        queue: nextQueue,
      });
    },
    [buildPayload, buildCompanion, promptValidation, setHydration, warnings],
  );

  const resetForNextDraft = () => {
    const current = form.getValues();
    const next: NewsForm = {
      ...current,
      user_prompt: "",
      title: "",
      artist_name: "",
      news_angle: "",
      caption: "",
      tags: "",
      negative_tags: "",
      lyrics: "",
      social_post_caption: "",
      social_hook_line: "",
      social_hashtags: "",
      art_prompt: "",
      video_prompt: "",
      genre_execution_contract: {},
      lyric_technique_report: {},
      lora_selection_reason: "",
      performance_notes: "",
      strict_completion_notes: "",
      payload_gate_status: "",
      payload_gate_passed: false,
      payload_gate_blocking_issues: [],
      payload_quality_gate: {},
      rap_quality_report: {},
      rap_rewrite_status: "",
      rap_blocking_issues: [],
      rap_strengths: [],
      rap_revision_focus: [],
    };
    form.reset(next);
    draftState.saveNow(next);
    setEditingQueueIndex(-1);
    setPasteBlocks(MODE, [{ label: "Huidige wizard JSON", content: "" }]);
  };

  const addCurrentToQueue = () => {
    const entry = buildDraftPayload();
    const nextQueue = [...queue];
    if (editingQueueIndex >= 0 && editingQueueIndex < nextQueue.length) {
      nextQueue[editingQueueIndex] = entry;
    } else {
      nextQueue.push(entry);
    }
    syncQueueState(nextQueue);
    resetForNextDraft();
  };

  const queueItemsForSubmit = React.useMemo(() => {
    const current = buildDraftPayload();
    if (editingQueueIndex >= 0) {
      return queue.map((item, index) => (index === editingQueueIndex ? current : item));
    }
    const hasMeaningfulDraft =
      Boolean((values.user_prompt ?? "").trim()) ||
      Boolean((values.title ?? "").trim()) ||
      Boolean((values.news_angle ?? "").trim()) ||
      Boolean((values.caption ?? "").trim()) ||
      Boolean((values.lyrics ?? "").trim());
    return hasMeaningfulDraft ? [...queue, current] : queue;
  }, [buildDraftPayload, editingQueueIndex, queue, values.caption, values.lyrics, values.news_angle, values.title, values.user_prompt]);

  const hydrate = (payload: Record<string, unknown>) => {
    const social = (payload.social_pack as Record<string, unknown> | undefined) ?? {};
    const merged: Partial<NewsForm> = {};
    for (const k of Object.keys(form.getValues())) {
      if (k in payload) {
        // @ts-expect-error dynamic
        merged[k] = payload[k];
      }
    }
    if (typeof social.post_caption === "string") merged.social_post_caption = social.post_caption;
    if (typeof social.hook_line === "string") merged.social_hook_line = social.hook_line;
    if (Array.isArray(social.hashtags)) merged.social_hashtags = social.hashtags.join(" ");
    const next = { ...form.getValues(), ...merged };
    form.reset(next);
    draftState.saveNow(next);
    return next;
  };

  const handleFinish = () => {
    const queued = queueItemsForSubmit;
    if (queued.length > 1) {
      void (async () => {
        try {
          const resp = await startSongBatchJob({
            batch_title: values.title || "Queued news songs",
            stop_on_error: false,
            songs: queued.map((item) => stripPromptCompanion(item)),
          });
          if (!resp.success || !resp.job_id) {
            throw new Error(resp.error || "News queue starten mislukt");
          }
          setStep(999);
        } catch (error) {
          console.error(error);
        }
      })();
      return;
    }
    void generation.start(buildPayload());
  };

  const audioUrl =
    firstGenerationAudioUrl(lastResult) ||
    (typeof lastResult?.audio === "string"
      ? `data:audio/wav;base64,${lastResult.audio}`
      : undefined);

  const steps: WizardStepDef[] = [
    {
      key: "ai",
      title: "Nieuws-prompt",
      description:
        "Plak een kop, link of beschrijving van het nieuws en de hoek die je wilt nemen. AI plant een satirische track in NL.",
      isValid: (values.user_prompt ?? "").trim().length >= 10 && !aiPromptPending,
      render: () => (
        <AIPromptStep
          mode="news"
          placeholder="Bijv. 'Kabinet valt over stikstof — maak er een drill-report van'"
          examples={[
            "Voetbal-rel: KNVB schorst trainer voor 5 wedstrijden — funny rap",
            "Energieprijzen stijgen weer — pop story met cynische ondertoon",
            "Nieuwe AI-wet wordt aangenomen — club banger 130 bpm",
          ]}
          onPendingChange={setAiPromptPending}
          onHydrated={(payload) => {
            const merged = hydrate(payload);
            const t =
              (payload.user_prompt as string | undefined) ??
              (payload.news_angle as string | undefined) ??
              "";
            if (t) {
              const withPrompt = { ...merged, user_prompt: t };
              form.reset(withPrompt);
              draftState.saveNow(withPrompt);
            }
          }}
          onManualApply={() => {
            setStep(1);
          }}
        />
      ),
    },
    {
      key: "angle",
      title: "Hoek & toon",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Identiteit">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label>Titel</Label>
                <Input {...form.register("title")} />
              </div>
              <div className="space-y-1.5">
                <Label>Artiest</Label>
                <Input {...form.register("artist_name")} />
              </div>
            </div>
          </FieldGroup>
          <FieldGroup title="Hoek & satire-modus">
            <div className="space-y-1.5">
              <Label>News angle</Label>
              <Textarea rows={3} {...form.register("news_angle")} placeholder="Welke kant draaien we het op?" />
            </div>
            <div className="space-y-1.5">
              <Label>Satire-modus</Label>
              <Controller
                control={form.control}
                name="satire_mode"
                render={({ field }) => (
                  <Select value={field.value} onValueChange={field.onChange}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {SATIRE_MODES.map(([id, label]) => (
                        <SelectItem key={id} value={id}>{label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              />
            </div>
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "lyrics",
      title: "Lyrics",
      description:
        "Voorgestelde tekst — pas aan of gebruik tags zoals [Verse] / [Chorus] voor structuur.",
      isValid: true,
      render: () => (
        <FieldGroup>
          <Textarea
            rows={16}
            className="font-mono text-xs leading-relaxed"
            {...form.register("lyrics")}
            placeholder={"[Verse 1]\n…"}
          />
        </FieldGroup>
      ),
    },
    {
      key: "social",
      title: "Social pack",
      description: "Extra metadata voor social media (caption, hook-line, hashtags).",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Post">
            <div className="space-y-1.5">
              <Label>Post caption</Label>
              <Textarea rows={3} {...form.register("social_post_caption")} />
            </div>
            <div className="space-y-1.5">
              <Label>Hook-line</Label>
              <Input {...form.register("social_hook_line")} />
            </div>
            <div className="space-y-1.5">
              <Label className="flex items-center gap-1.5">
                <Hash className="size-3.5" /> Hashtags
              </Label>
              <Input placeholder="#stikstof #drill #nu" {...form.register("social_hashtags")} />
            </div>
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "render",
      title: "Render-instellingen",
      isValid: true,
      render: () => (
        <div className="space-y-4">
          <FieldGroup title="Lengte & sound">
            <div className="space-y-3">
              <div className="flex items-baseline justify-between">
                <Label>Duur</Label>
                <span className="font-mono text-sm">{formatDuration(values.duration)}</span>
              </div>
              <Controller
                control={form.control}
                name="duration"
                render={({ field }) => (
                  <Slider value={[field.value]} min={20} max={300} step={5} onValueChange={(v) => field.onChange(v[0])} />
                )}
              />
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="space-y-1.5">
                <Label>BPM</Label>
                <Input type="number" placeholder="auto" {...form.register("bpm", { valueAsNumber: true })} />
              </div>
              <div className="space-y-1.5">
                <Label>Key</Label>
                <Input {...form.register("key_scale")} />
              </div>
              <div className="space-y-1.5">
                <Label>Taal</Label>
                <Controller
                  control={form.control}
                  name="vocal_language"
                  render={({ field }) => (
                    <Select value={field.value || "unknown"} onValueChange={field.onChange}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {ACE_STEP_LANGUAGE_OPTIONS.map(([code, label]) => (
                          <SelectItem key={code} value={code}>{label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>
            </div>
          </FieldGroup>
          <FieldGroup title="Audio backend">
            <AudioBackendSelector
              value={values.audio_backend}
              onChange={(value) => form.setValue("audio_backend", value, { shouldValidate: true })}
            />
          </FieldGroup>
          <FieldGroup title="Tags">
            <AudioStyleSelector
              value={values.style_profile}
              onChange={(value) => form.setValue("style_profile", value, { shouldValidate: true })}
            />
            <Controller
              control={form.control}
              name="tags"
              render={({ field }) => (
                <TagInput value={field.value} onChange={field.onChange} />
              )}
            />
            <Controller
              control={form.control}
              name="negative_tags"
              render={({ field }) => (
                <TagInput value={field.value} onChange={field.onChange} variant="negative" />
              )}
            />
          </FieldGroup>
          <FieldGroup title="LoRA" description="Optioneel: kies een getrainde PEFT LoRA voor deze track.">
            <LoraSelector value={values} onChange={setLoraSelection} />
          </FieldGroup>
          <AutomationFields control={form.control} register={form.register} values={values} />
          <FieldGroup title="Variaties" description="Zelfde nieuws-song, andere seed.">
            <div className="flex items-center gap-3">
              <Controller
                control={form.control}
                name="batch_size"
                render={({ field }) => (
                  <Slider value={[field.value]} min={1} max={8} step={1} onValueChange={(value) => field.onChange(value[0] ?? 1)} />
                )}
              />
              <span className="w-8 text-right font-mono text-xs">{values.batch_size}</span>
            </div>
          </FieldGroup>
        </div>
      ),
    },
    {
      key: "review",
      title: "Review & genereer",
      isValid: form.formState.isValid && reviewBlockingIssues.length === 0,
      render: () => (
        <div className="space-y-4">
          <ReviewStep
            payload={buildDraftPayload()}
            warnings={warnings}
            blockingIssues={reviewBlockingIssues}
            queueSummary={{
              queuedItems: queue.length,
              totalRenders: queueItemsForSubmit.reduce((sum, item) => sum + Number(item.batch_size || 1), 0),
              label: "news songs",
            }}
            companion={buildCompanion()}
            primaryFields={[
              { key: "title", label: "Titel" },
              { key: "satire_mode", label: "Satire-modus" },
              { key: "duration", label: "Duur", format: (v) => formatDuration(Number(v) || 0) },
              { key: "song_model", label: "Model" },
              { key: "audio_backend", label: "Backend", format: audioBackendLabel },
              { key: "lora_adapter_name", label: "LoRA" },
              { key: "lora_trigger_tag", label: "LoRA trigger" },
              { key: "vocal_language", label: "Taal" },
              { key: "batch_size", label: "Variaties" },
              { key: "social_hook_line", label: "Hook" },
              { key: "tags", label: "Tags" },
            ]}
          />
          <MusicQueueEditor
            items={queue.map((item, index) => summarizeQueueEntry(item, `News song ${index + 1}`))}
            activeIndex={editingQueueIndex}
            onSelect={(index) => {
              const selected = queue[index];
              if (!selected) return;
              hydrate(selected);
              setEditingQueueIndex(index);
              setPasteBlocks(MODE, [
                { label: "Huidige wizard JSON", content: JSON.stringify(selected, null, 2) },
              ]);
            }}
            onRemove={(index) => {
              const nextQueue = queue.filter((_, itemIndex) => itemIndex !== index);
              syncQueueState(nextQueue);
              if (editingQueueIndex === index) setEditingQueueIndex(-1);
            }}
            onDuplicate={(index) => {
              const nextQueue = [...queue];
              nextQueue.splice(index + 1, 0, { ...queue[index] });
              syncQueueState(nextQueue);
            }}
            onMove={(index, direction) => {
              const target = index + direction;
              if (target < 0 || target >= queue.length) return;
              const nextQueue = [...queue];
              const [item] = nextQueue.splice(index, 1);
              nextQueue.splice(target, 0, item);
              syncQueueState(nextQueue);
            }}
          />
          <RenderInsightPanel payload={buildPayload()} warnings={warnings} />
          {(generation.jobId || generation.isSubmitting) && (
            <GenerationJobStatus
              job={generation.activeJob}
              jobId={generation.jobId}
              onOpen={generation.openActiveJob}
            />
          )}
        </div>
      ),
    },
    {
      key: "result",
      title: "Resultaat",
      isValid: true,
      hidden: !lastResult,
      render: () => {
        if (!lastResult) return null;
        const resultId = (lastResult.result_id as string) || (lastResult.id as string);
        const title = (lastResult.title as string) || values.title || "News track";
        const artist = (lastResult.artist_name as string) || values.artist_name;
        return (
          <div className="space-y-4">
            <GenerationAudioList result={lastResult} title={title} artist={artist} />
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex flex-wrap gap-2"
            >
              <Button variant="outline" onClick={() => navigate("/library")} className="gap-2">
                <Music4 className="size-4" /> Open library
              </Button>
              {audioUrl && (
                <Button
                  variant="outline"
                  onClick={() => navigate("/wizard/video", {
                    state: {
                      audio_url: audioUrl,
                      title,
                      artist_name: artist,
                      prompt: String(lastResult.caption || values.caption || values.tags || title || ""),
                      target_type: "song",
                      target_id: resultId || title,
                    },
                  })}
                  className="gap-2"
                >
                  <Video className="size-4" /> Create video
                </Button>
              )}
              <Button variant="ghost" onClick={() => { form.reset(newsDefaults); draftState.clear(); setStep(0); }}>Volgende nieuwsbericht</Button>
            </motion.div>
          </div>
        );
      },
    },
  ];

  return (
    <WizardShell
      title="News wizard"
      subtitle="Van actueel nieuws naar satirische NL-track in een paar minuten."
      steps={steps}
      step={step}
      onStepChange={setStep}
      onFinish={handleFinish}
      secondaryFinishAction={{
        label: "Nog een toevoegen",
        onClick: addCurrentToQueue,
        disabled: !form.formState.isValid || reviewBlockingIssues.length > 0,
      }}
      isFinishing={generation.isSubmitting || generation.isRunning}
      finishLabel={generation.isSubmitting || generation.isRunning ? "Renderen…" : "Nu genereren"}
    />
  );
}
