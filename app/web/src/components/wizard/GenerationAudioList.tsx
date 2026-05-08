import * as React from "react";

import { WaveformPlayer } from "@/components/audio/WaveformPlayer";

type JsonRecord = Record<string, unknown>;

function asRecord(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function text(value: unknown, fallback = ""): string {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

export function firstGenerationAudioUrl(result: unknown): string {
  const record = asRecord(result);
  const firstAudio = asRecord(asArray(record.audios)[0]);
  return text(
    firstAudio.audio_url ||
      firstAudio.download_url ||
      firstAudio.library_url ||
      record.audio_url ||
      record.download_url,
    "",
  );
}

export function GenerationAudioList({
  result,
  title,
  artist,
  className,
}: {
  result: unknown;
  title?: string;
  artist?: string;
  className?: string;
}) {
  const record = asRecord(result);
  const audios = asArray(record.audios)
    .map(asRecord)
    .filter((item) => text(item.audio_url || item.download_url || item.library_url, ""));
  const fallbackUrl = firstGenerationAudioUrl(record);
  const items = audios.length
    ? audios
    : fallbackUrl
      ? [{ audio_url: fallbackUrl }]
      : [];

  if (!items.length) return null;

  return (
    <div className={className ?? "space-y-3"}>
      {items.map((audio, index) => {
        const src = text(audio.audio_url || audio.download_url || audio.library_url);
        const audioTitle =
          text(audio.title, "") ||
          (items.length > 1 ? `${title || text(record.title, "Track")} · take ${index + 1}` : title || text(record.title, "Track"));
        return (
          <WaveformPlayer
            key={`${src}-${index}`}
            src={src}
            title={audioTitle}
            artist={text(audio.artist_name, artist || text(record.artist_name, ""))}
            metadata={{
              model: audio.song_model ?? record.song_model ?? record.active_song_model,
              quality: record.quality_profile,
              duration: record.duration,
              bpm: record.bpm,
              key: record.key_scale,
              seed: audio.seed ?? record.seed,
              resultId: audio.result_id ?? record.result_id,
            }}
          />
        );
      })}
    </div>
  );
}
