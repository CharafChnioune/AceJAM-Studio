import * as React from "react";
import { useNavigate } from "react-router-dom";
import { useForm, Controller } from "react-hook-form";
import { motion } from "framer-motion";
import { Newspaper, Music4, Hash, Video } from "lucide-react";

import { WizardShell, FieldGroup, type WizardStepDef } from "@/components/wizard/WizardShell";
import { AIPromptStep } from "@/components/wizard/AIPromptStep";
import { ReviewStep } from "@/components/wizard/ReviewStep";
import { GenerationJobStatus } from "@/components/wizard/GenerationJobStatus";
import { GenerationAudioList, firstGenerationAudioUrl } from "@/components/wizard/GenerationAudioList";
import { LoraSelector } from "@/components/wizard/LoraSelector";
import { RenderInsightPanel } from "@/components/wizard/RenderInsightPanel";
import { TagInput } from "@/components/wizard/TagInput";
import { AutomationFields } from "@/components/wizard/AutomationFields";
import { AudioStyleSelector } from "@/components/wizard/AudioStyleSelector";
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
import { formatDuration } from "@/lib/utils";

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
  use_lora: boolean;
  lora_adapter_path: string;
  lora_adapter_name: string;
  use_lora_trigger: boolean;
  lora_trigger_tag: string;
  lora_scale: number;
  adapter_model_variant: string;
  adapter_song_model: string;
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
  const lastResult = useWizardStore((s) => s.lastResult[MODE]);
  const warnings = useWizardStore((s) => s.warnings[MODE]) ?? [];
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
      use_lora: false,
      lora_adapter_path: "",
      lora_adapter_name: "",
      use_lora_trigger: false,
      lora_trigger_tag: "",
      lora_scale: DEFAULT_LORA_SCALE,
      adapter_model_variant: "",
      adapter_song_model: "",
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
  const values = form.watch();
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
    form.setValue("lora_scale", selection.lora_scale, { shouldValidate: true });
    form.setValue("adapter_model_variant", selection.adapter_model_variant, { shouldValidate: true });
    form.setValue("adapter_song_model", selection.adapter_song_model, { shouldValidate: true });
    if (selection.use_lora && selection.adapter_song_model) {
      form.setValue("song_model", selection.adapter_song_model, { shouldValidate: true });
    }
  };

  const buildPayload = () => {
    const v = form.getValues();
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
                <Input {...form.register("vocal_language")} />
              </div>
            </div>
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
        </div>
      ),
    },
    {
      key: "review",
      title: "Review & genereer",
      isValid: form.formState.isValid,
      render: () => (
        <div className="space-y-4">
          <ReviewStep
            payload={buildPayload()}
            warnings={warnings}
            primaryFields={[
              { key: "title", label: "Titel" },
              { key: "satire_mode", label: "Satire-modus" },
              { key: "duration", label: "Duur", format: (v) => formatDuration(Number(v) || 0) },
              { key: "song_model", label: "Model" },
              { key: "lora_adapter_name", label: "LoRA" },
              { key: "lora_trigger_tag", label: "LoRA trigger" },
              { key: "vocal_language", label: "Taal" },
              { key: "social_hook_line", label: "Hook" },
              { key: "tags", label: "Tags" },
            ]}
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
      onFinish={() => void generation.start(buildPayload())}
      isFinishing={generation.isSubmitting || generation.isRunning}
      finishLabel={generation.isSubmitting || generation.isRunning ? "Renderen…" : "Genereer track"}
    />
  );
}
