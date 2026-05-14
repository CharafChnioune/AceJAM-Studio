export type AudioBackend = "mlx" | "mps_torch";

export const DEFAULT_AUDIO_BACKEND: AudioBackend = "mps_torch";

export function normalizeAudioBackend(value: unknown): AudioBackend {
  const raw = String(value || "").trim().toLowerCase().replace("-", "_");
  if (raw === "mlx" || raw === "native_mlx" || raw === "mlx_dit" || raw === "mlx_audio") {
    return "mlx";
  }
  if (raw === "mps" || raw === "mps_torch" || raw === "torch" || raw === "pytorch" || raw === "pt") {
    return "mps_torch";
  }
  return DEFAULT_AUDIO_BACKEND;
}

export function useMlxDitForAudioBackend(value: unknown): boolean {
  return normalizeAudioBackend(value) === "mlx";
}

export function audioBackendLabel(value: unknown): string {
  return normalizeAudioBackend(value) === "mlx" ? "MLX" : "MPS/Torch";
}
