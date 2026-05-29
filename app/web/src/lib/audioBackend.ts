export type AudioBackend = "mlx";

export const DEFAULT_AUDIO_BACKEND: AudioBackend = "mlx";

export function normalizeAudioBackend(value: unknown): AudioBackend {
  const raw = String(value || "").trim().toLowerCase().replace("-", "_");
  if (raw) return "mlx";
  return DEFAULT_AUDIO_BACKEND;
}

export function useMlxDitForAudioBackend(value: unknown): boolean {
  return normalizeAudioBackend(value) === "mlx";
}

export function audioBackendLabel(value: unknown): string {
  normalizeAudioBackend(value);
  return "MLX";
}
