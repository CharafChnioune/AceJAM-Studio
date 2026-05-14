import * as React from "react";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  ACE_STEP_ADVANCED_DEFAULTS,
  OFFICIAL_AUDIO_FORMAT_OPTIONS,
  type AceStepAdvancedField,
  type AceStepAdvancedValues,
} from "@/lib/aceStepSettings";

type AdvancedKey =
  | AceStepAdvancedField
  | "audio_format"
  | "batch_size"
  | "inference_steps"
  | "guidance_scale"
  | "shift"
  | "seed";

interface AceStepAdvancedSettingsProps {
  values: AceStepAdvancedValues;
  onChange: (key: AdvancedKey, value: unknown) => void;
  showSourceControls?: boolean;
}

const SELECTS = {
  infer_method: [
    ["ode", "ODE"],
    ["sde", "SDE"],
  ],
  sampler_mode: [
    ["euler", "Euler"],
    ["heun", "Heun"],
  ],
  dcw_mode: [
    ["low", "Low band"],
    ["high", "High band"],
    ["double", "Double"],
    ["pix", "Pixel"],
  ],
  dcw_wavelet: [
    ["haar", "Haar"],
    ["db4", "Daubechies 4"],
    ["sym8", "Symlet 8"],
    ["coif1", "Coiflet 1"],
    ["bior2.2", "Biorthogonal 2.2"],
  ],
  mp3_bitrate: [
    ["128k", "128k"],
    ["192k", "192k"],
    ["256k", "256k"],
    ["320k", "320k"],
  ],
  mp3_sample_rate: [
    ["44100", "44.1 kHz"],
    ["48000", "48 kHz"],
  ],
  device: [
    ["auto", "Auto"],
    ["mps", "Apple MPS"],
    ["cuda", "CUDA"],
    ["cpu", "CPU"],
  ],
  dtype: [
    ["auto", "Auto"],
    ["float32", "Float32"],
    ["float16", "Float16"],
    ["bfloat16", "BFloat16"],
  ],
  use_flash_attention: [
    ["auto", "Auto"],
    ["true", "Force on"],
    ["false", "Force off"],
  ],
  vae_checkpoint: [
    ["official", "Official VAE"],
    ["scragvae", "ScragVAE"],
  ],
  chunk_mask_mode: [
    ["auto", "Auto"],
    ["explicit", "Explicit mask"],
  ],
  repaint_mode: [
    ["balanced", "Balanced"],
    ["conservative", "Conservative"],
    ["aggressive", "Aggressive"],
  ],
  track_name: [
    ["", "Auto"],
    ["vocals", "Vocals"],
    ["drums", "Drums"],
    ["bass", "Bass"],
    ["other", "Other"],
    ["guitar", "Guitar"],
    ["piano", "Piano"],
    ["strings", "Strings"],
  ],
} as const;

function fallback(values: AceStepAdvancedValues, key: AdvancedKey, fallbackValue?: unknown) {
  const current = values[key as keyof AceStepAdvancedValues];
  if (current !== undefined && current !== null) return current;
  if (fallbackValue !== undefined) return fallbackValue;
  return ACE_STEP_ADVANCED_DEFAULTS[key as AceStepAdvancedField];
}

function numberValue(values: AceStepAdvancedValues, key: AdvancedKey, fallbackValue: number) {
  const value = Number(fallback(values, key, fallbackValue));
  return Number.isFinite(value) ? value : fallbackValue;
}

function textValue(values: AceStepAdvancedValues, key: AdvancedKey, fallbackValue = "") {
  const value = fallback(values, key, fallbackValue);
  return value === undefined || value === null ? "" : String(value);
}

function boolValue(values: AceStepAdvancedValues, key: AdvancedKey, fallbackValue = false) {
  const value = fallback(values, key, fallbackValue);
  return Boolean(value);
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <Label>{label}</Label>
      {children}
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-3 rounded-md border bg-background/35 p-3">
      <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </h4>
      {children}
    </div>
  );
}

function NumberInput({
  label,
  values,
  field,
  onChange,
  min,
  max,
  step = 1,
  fallbackValue,
}: {
  label: string;
  values: AceStepAdvancedValues;
  field: AdvancedKey;
  onChange: (key: AdvancedKey, value: unknown) => void;
  min?: number;
  max?: number;
  step?: number;
  fallbackValue: number;
}) {
  return (
    <Field label={label}>
      <Input
        type="number"
        min={min}
        max={max}
        step={step}
        value={numberValue(values, field, fallbackValue)}
        onChange={(event) => onChange(field, Number(event.target.value))}
      />
    </Field>
  );
}

function SelectField({
  label,
  values,
  field,
  options,
  onChange,
  fallbackValue,
}: {
  label: string;
  values: AceStepAdvancedValues;
  field: AdvancedKey;
  options: readonly (readonly [string, string])[];
  onChange: (key: AdvancedKey, value: unknown) => void;
  fallbackValue: string;
}) {
  return (
    <Field label={label}>
      <Select
        value={textValue(values, field, fallbackValue) || "__empty"}
        onValueChange={(value) => {
          const parsed =
            field === "mp3_sample_rate" ? Number(value) : value === "__empty" ? "" : value;
          onChange(field, parsed);
        }}
      >
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map(([value, label]) => (
            <SelectItem key={value || "__empty"} value={value || "__empty"}>
              {label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </Field>
  );
}

function SwitchField({
  label,
  values,
  field,
  onChange,
  fallbackValue,
}: {
  label: string;
  values: AceStepAdvancedValues;
  field: AdvancedKey;
  onChange: (key: AdvancedKey, value: unknown) => void;
  fallbackValue: boolean;
}) {
  return (
    <label className="flex min-h-10 items-center justify-between gap-3 rounded-md border bg-background/35 px-3 py-2 text-sm">
      <span>{label}</span>
      <Switch
        checked={boolValue(values, field, fallbackValue)}
        onCheckedChange={(checked) => onChange(field, checked)}
      />
    </label>
  );
}

function SliderField({
  label,
  values,
  field,
  onChange,
  min,
  max,
  step,
  fallbackValue,
}: {
  label: string;
  values: AceStepAdvancedValues;
  field: AdvancedKey;
  onChange: (key: AdvancedKey, value: unknown) => void;
  min: number;
  max: number;
  step: number;
  fallbackValue: number;
}) {
  const value = numberValue(values, field, fallbackValue);
  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <Label>{label}</Label>
        <span className="font-mono text-xs">{Number.isInteger(value) ? value : value.toFixed(3)}</span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(next) => onChange(field, next[0] ?? fallbackValue)}
      />
    </div>
  );
}

export function AceStepAdvancedSettings({
  values,
  onChange,
  showSourceControls = true,
}: AceStepAdvancedSettingsProps) {
  return (
    <div className="space-y-4">
      <Section title="Diffusion">
        <div className="grid gap-3 md:grid-cols-2">
          <SelectField label="Infer method" values={values} field="infer_method" options={SELECTS.infer_method} onChange={onChange} fallbackValue="ode" />
          <SelectField label="Sampler" values={values} field="sampler_mode" options={SELECTS.sampler_mode} onChange={onChange} fallbackValue="euler" />
          <SwitchField label="Adaptive dual guidance" values={values} field="use_adg" onChange={onChange} fallbackValue={false} />
          <NumberInput label="CFG interval start" values={values} field="cfg_interval_start" onChange={onChange} min={0} max={1} step={0.01} fallbackValue={0} />
          <NumberInput label="CFG interval end" values={values} field="cfg_interval_end" onChange={onChange} min={0} max={1} step={0.01} fallbackValue={1} />
          <Field label="Custom timesteps">
            <Input
              placeholder="0.97, 0.76, 0.5, 0.0"
              value={textValue(values, "timesteps")}
              onChange={(event) => onChange("timesteps", event.target.value)}
            />
          </Field>
        </div>
      </Section>

      <Section title="DCW wavelet correction">
        <div className="grid gap-3 md:grid-cols-2">
          <SwitchField label="DCW aan" values={values} field="dcw_enabled" onChange={onChange} fallbackValue />
          <SelectField label="DCW mode" values={values} field="dcw_mode" options={SELECTS.dcw_mode} onChange={onChange} fallbackValue="double" />
          <SliderField label="Low scaler" values={values} field="dcw_scaler" onChange={onChange} min={0} max={0.1} step={0.005} fallbackValue={0.05} />
          <SliderField label="High scaler" values={values} field="dcw_high_scaler" onChange={onChange} min={0} max={0.1} step={0.005} fallbackValue={0.02} />
          <SelectField label="Wavelet" values={values} field="dcw_wavelet" options={SELECTS.dcw_wavelet} onChange={onChange} fallbackValue="haar" />
        </div>
      </Section>

      <Section title="Output & post">
        <div className="grid gap-3 md:grid-cols-2">
          <SelectField label="Audio format" values={values} field="audio_format" options={OFFICIAL_AUDIO_FORMAT_OPTIONS} onChange={onChange} fallbackValue="wav32" />
          <SelectField label="MP3 bitrate" values={values} field="mp3_bitrate" options={SELECTS.mp3_bitrate} onChange={onChange} fallbackValue="128k" />
          <SelectField label="Sample rate" values={values} field="mp3_sample_rate" options={SELECTS.mp3_sample_rate} onChange={onChange} fallbackValue="48000" />
          <SwitchField label="Auto score" values={values} field="auto_score" onChange={onChange} fallbackValue={false} />
          <SwitchField label="Auto LRC" values={values} field="auto_lrc" onChange={onChange} fallbackValue={false} />
          <SwitchField label="Audio codes teruggeven" values={values} field="return_audio_codes" onChange={onChange} fallbackValue={false} />
          <SwitchField label="Opslaan in library" values={values} field="save_to_library" onChange={onChange} fallbackValue={false} />
          <SwitchField label="Normalisatie" values={values} field="enable_normalization" onChange={onChange} fallbackValue />
          <NumberInput label="Normalisatie dB" values={values} field="normalization_db" onChange={onChange} min={-24} max={0} step={0.5} fallbackValue={-1} />
          <NumberInput label="Fade in" values={values} field="fade_in_duration" onChange={onChange} min={0} max={20} step={0.1} fallbackValue={0} />
          <NumberInput label="Fade out" values={values} field="fade_out_duration" onChange={onChange} min={0} max={20} step={0.1} fallbackValue={0} />
        </div>
      </Section>

      <Section title="Latent, retake & runtime">
        <div className="grid gap-3 md:grid-cols-2">
          <Field label="Retake seed">
            <Input value={textValue(values, "retake_seed")} onChange={(event) => onChange("retake_seed", event.target.value)} />
          </Field>
          <SliderField label="Retake variance" values={values} field="retake_variance" onChange={onChange} min={0} max={1} step={0.01} fallbackValue={0} />
          <NumberInput label="Latent shift" values={values} field="latent_shift" onChange={onChange} min={-2} max={2} step={0.05} fallbackValue={0} />
          <NumberInput label="Latent rescale" values={values} field="latent_rescale" onChange={onChange} min={0.1} max={3} step={0.05} fallbackValue={1} />
          <SwitchField label="Tiled decode" values={values} field="use_tiled_decode" onChange={onChange} fallbackValue />
          <SwitchField label="Caption is al geformatteerd" values={values} field="is_format_caption" onChange={onChange} fallbackValue={false} />
          <SelectField label="Device" values={values} field="device" options={SELECTS.device} onChange={onChange} fallbackValue="auto" />
          <SelectField label="Dtype" values={values} field="dtype" options={SELECTS.dtype} onChange={onChange} fallbackValue="auto" />
          <SelectField label="Flash attention" values={values} field="use_flash_attention" options={SELECTS.use_flash_attention} onChange={onChange} fallbackValue="auto" />
          <SelectField label="VAE checkpoint" values={values} field="vae_checkpoint" options={SELECTS.vae_checkpoint} onChange={onChange} fallbackValue="official" />
          <SwitchField label="Compile model" values={values} field="compile_model" onChange={onChange} fallbackValue={false} />
          <SwitchField label="Offload naar CPU" values={values} field="offload_to_cpu" onChange={onChange} fallbackValue={false} />
          <SwitchField label="Offload DiT naar CPU" values={values} field="offload_dit_to_cpu" onChange={onChange} fallbackValue={false} />
        </div>
      </Section>

      {showSourceControls && (
        <Section title="Source, repaint, extract & complete">
          <div className="grid gap-3 md:grid-cols-2">
            <Field label="Instruction">
              <Textarea rows={2} value={textValue(values, "instruction")} onChange={(event) => onChange("instruction", event.target.value)} />
            </Field>
            <Field label="Global caption">
              <Textarea rows={2} value={textValue(values, "global_caption")} onChange={(event) => onChange("global_caption", event.target.value)} />
            </Field>
            <Field label="Audio codes">
              <Textarea rows={2} value={textValue(values, "audio_codes")} onChange={(event) => onChange("audio_codes", event.target.value)} />
            </Field>
            <NumberInput label="Audio cover strength" values={values} field="audio_cover_strength" onChange={onChange} min={0} max={1} step={0.01} fallbackValue={1} />
            <NumberInput label="Cover noise strength" values={values} field="cover_noise_strength" onChange={onChange} min={0} max={1} step={0.01} fallbackValue={0} />
            <NumberInput label="Repainting start" values={values} field="repainting_start" onChange={onChange} min={0} step={0.1} fallbackValue={0} />
            <NumberInput label="Repainting end" values={values} field="repainting_end" onChange={onChange} step={0.1} fallbackValue={-1} />
            <SelectField label="Chunk mask" values={values} field="chunk_mask_mode" options={SELECTS.chunk_mask_mode} onChange={onChange} fallbackValue="auto" />
            <SelectField label="Repaint mode" values={values} field="repaint_mode" options={SELECTS.repaint_mode} onChange={onChange} fallbackValue="balanced" />
            <NumberInput label="Repaint strength" values={values} field="repaint_strength" onChange={onChange} min={0} max={1} step={0.01} fallbackValue={0.5} />
            <NumberInput label="Latent crossfade frames" values={values} field="repaint_latent_crossfade_frames" onChange={onChange} min={0} max={200} step={1} fallbackValue={10} />
            <NumberInput label="WAV crossfade sec" values={values} field="repaint_wav_crossfade_sec" onChange={onChange} min={0} max={20} step={0.1} fallbackValue={0} />
            <SwitchField label="Flow edit morph" values={values} field="flow_edit_morph" onChange={onChange} fallbackValue={false} />
            <NumberInput label="Flow N min" values={values} field="flow_edit_n_min" onChange={onChange} min={0} max={1} step={0.01} fallbackValue={0} />
            <NumberInput label="Flow N max" values={values} field="flow_edit_n_max" onChange={onChange} min={0} max={1} step={0.01} fallbackValue={1} />
            <NumberInput label="Flow N avg" values={values} field="flow_edit_n_avg" onChange={onChange} min={1} max={16} step={1} fallbackValue={1} />
            <SelectField label="Track name" values={values} field="track_name" options={SELECTS.track_name} onChange={onChange} fallbackValue="" />
            <Field label="Track classes">
              <Input
                placeholder="vocals, drums"
                value={Array.isArray(values.track_classes) ? values.track_classes.join(", ") : textValue(values, "track_classes")}
                onChange={(event) =>
                  onChange(
                    "track_classes",
                    event.target.value
                      .split(",")
                      .map((item) => item.trim())
                      .filter(Boolean),
                  )
                }
              />
            </Field>
            <Field label="Flow source caption">
              <Textarea rows={2} value={textValue(values, "flow_edit_source_caption")} onChange={(event) => onChange("flow_edit_source_caption", event.target.value)} />
            </Field>
            <Field label="Flow source lyrics">
              <Textarea rows={2} value={textValue(values, "flow_edit_source_lyrics")} onChange={(event) => onChange("flow_edit_source_lyrics", event.target.value)} />
            </Field>
          </div>
        </Section>
      )}
    </div>
  );
}
