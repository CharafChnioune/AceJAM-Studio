import type { Control, FieldValues, UseFormRegister } from "react-hook-form";
import { Controller } from "react-hook-form";
import { ImagePlus, Video } from "lucide-react";

import { FieldGroup } from "@/components/wizard/WizardShell";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";

interface AutomationValues {
  auto_song_art?: boolean;
  auto_album_art?: boolean;
  auto_video_clip?: boolean;
}

export function AutomationFields<T extends FieldValues>({
  control,
  register,
  values,
  albumContext,
}: {
  control: Control<T>;
  register: UseFormRegister<T>;
  values: AutomationValues;
  albumContext?: boolean;
}) {
  return (
    <FieldGroup
      title="After audio render"
      description="Maak optioneel direct artwork en een muxed muziekvideo die aan het resultaat gekoppeld worden."
    >
      <div className="grid gap-3 sm:grid-cols-3">
        <label className="flex items-center justify-between gap-3 rounded-md border bg-background/35 p-3 text-sm">
          <span className="flex items-center gap-2">
            <ImagePlus className="size-4" />
            Song art
          </span>
          <Controller
            control={control}
            name={"auto_song_art" as never}
            render={({ field }) => <Switch checked={Boolean(field.value)} onCheckedChange={field.onChange} />}
          />
        </label>
        <label className="flex items-center justify-between gap-3 rounded-md border bg-background/35 p-3 text-sm">
          <span className="flex items-center gap-2">
            <ImagePlus className="size-4" />
            Album art
          </span>
          <Controller
            control={control}
            name={"auto_album_art" as never}
            render={({ field }) => (
              <Switch checked={Boolean(field.value)} onCheckedChange={field.onChange} disabled={!albumContext} />
            )}
          />
        </label>
        <label className="flex items-center justify-between gap-3 rounded-md border bg-background/35 p-3 text-sm">
          <span className="flex items-center gap-2">
            <Video className="size-4" />
            Video clip
          </span>
          <Controller
            control={control}
            name={"auto_video_clip" as never}
            render={({ field }) => <Switch checked={Boolean(field.value)} onCheckedChange={field.onChange} />}
          />
        </label>
      </div>
      {(values.auto_song_art || values.auto_album_art) && (
        <div className="space-y-1.5">
          <Label>Art prompt override</Label>
          <Textarea
            rows={2}
            placeholder="Leeg = MLX Media bouwt prompt uit titel, artiest, genre en mood."
            {...register("art_prompt" as never)}
          />
        </div>
      )}
      {values.auto_video_clip && (
        <div className="space-y-1.5">
          <Label>Video prompt override</Label>
          <Textarea
            rows={2}
            placeholder="Leeg = kleine draft muziekvideo uit song art/album art plus source audio."
            {...register("video_prompt" as never)}
          />
        </div>
      )}
    </FieldGroup>
  );
}
