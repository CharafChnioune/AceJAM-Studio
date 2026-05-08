import * as React from "react";
import { useQuery } from "@tanstack/react-query";

import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getAudioStyleProfiles, type AudioStyleProfile } from "@/lib/api";

const FALLBACK_STYLE_PROFILES: AudioStyleProfile[] = [
  {
    key: "auto",
    style_profile: "auto",
    label: "Auto",
    caption_tags: "Infer style from caption/tags.",
    lyrics_section_tags: {},
  },
  {
    key: "rap",
    style_profile: "rap",
    label: "Rap / Hip-hop",
    caption_tags: "rap, hip hop, rhythmic spoken-word vocal, clear rap flow, deep bass, hard drums",
    lyrics_section_tags: { verse: "rap, rhythmic spoken flow", chorus: "rap hook" },
  },
  {
    key: "pop",
    style_profile: "pop",
    label: "Pop",
    caption_tags: "modern pop groove, bright hook, clean lead vocal, radio-ready drums",
    lyrics_section_tags: { verse: "clean pop vocal", chorus: "bright pop hook" },
  },
  {
    key: "rnb",
    style_profile: "rnb",
    label: "Soul / R&B",
    caption_tags: "smooth rnb groove, warm keys, clean intimate lead vocal, soft harmonies",
    lyrics_section_tags: { verse: "smooth rnb vocal", chorus: "soulful rnb hook" },
  },
  {
    key: "rock",
    style_profile: "rock",
    label: "Rock",
    caption_tags: "driving rock drums, electric guitars, clear lead vocal, strong chorus",
    lyrics_section_tags: { verse: "rock lead vocal", chorus: "strong rock chorus" },
  },
  {
    key: "edm",
    style_profile: "edm",
    label: "EDM / Dance",
    caption_tags: "electronic dance beat, pulsing synth bass, clean vocal hook, club energy",
    lyrics_section_tags: { verse: "dance vocal", chorus: "club vocal hook" },
  },
  {
    key: "cinematic",
    style_profile: "cinematic",
    label: "Cinematic",
    caption_tags: "cinematic drums, wide strings, clear dramatic vocal, spacious arrangement",
    lyrics_section_tags: { verse: "dramatic vocal", chorus: "cinematic anthem" },
  },
  {
    key: "country",
    style_profile: "country",
    label: "Country / Folk",
    caption_tags: "warm acoustic guitars, steady country drums, clear heartfelt vocal",
    lyrics_section_tags: { verse: "country lead vocal", chorus: "heartfelt country hook" },
  },
];

function profileKey(profile: AudioStyleProfile) {
  return profile.key || profile.style_profile || "auto";
}

export function AudioStyleSelector({
  value,
  onChange,
}: {
  value?: string;
  onChange: (value: string) => void;
}) {
  const query = useQuery({
    queryKey: ["audio-style-profiles"],
    queryFn: getAudioStyleProfiles,
    staleTime: 10 * 60 * 1000,
  });
  const profiles = React.useMemo(() => {
    const fromApi = query.data?.profiles?.filter((item) => profileKey(item)) ?? [];
    return fromApi.length ? fromApi : FALLBACK_STYLE_PROFILES;
  }, [query.data?.profiles]);
  const selected =
    profiles.find((item) => profileKey(item) === (value || "auto")) ??
    profiles.find((item) => profileKey(item) === "auto") ??
    profiles[0];
  const sectionTags = selected?.lyrics_section_tags ?? {};
  return (
    <div className="space-y-2">
      <div className="grid gap-3 sm:grid-cols-[220px_1fr]">
        <div className="space-y-1.5">
          <Label>Genre / stijlprofiel</Label>
          <Select value={value || "auto"} onValueChange={onChange}>
            <SelectTrigger>
              <SelectValue placeholder="Kies stijl" />
            </SelectTrigger>
            <SelectContent>
              {profiles.map((profile) => (
                <SelectItem key={profileKey(profile)} value={profileKey(profile)}>
                  {profile.label || profileKey(profile)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="min-w-0 space-y-1.5">
          <Label>Caption tags</Label>
          <p className="rounded-md border bg-background/35 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
            {selected?.caption_tags || "Auto gebruikt je caption/tags zonder extra genre te forceren."}
          </p>
        </div>
      </div>
      {Object.keys(sectionTags).length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(sectionTags).map(([section, tags]) => (
            <Badge key={section} variant="outline" className="text-[10px]">
              {section}: {String(tags)}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}
