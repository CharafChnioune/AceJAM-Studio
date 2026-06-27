export interface AudioDownloadVariants {
  master?: string;
  whatsapp?: string;
}

type VariantLike = AudioDownloadVariants | Record<string, unknown> | null | undefined;

export function audioDownloadVariants(value: VariantLike): AudioDownloadVariants {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const record = value as Record<string, unknown>;
  return {
    master: typeof record.master === "string" ? record.master : undefined,
    whatsapp: typeof record.whatsapp === "string" ? record.whatsapp : undefined,
  };
}

export function masterAudioDownloadUrl(src: string, downloads?: VariantLike): string {
  return audioDownloadVariants(downloads).master || src;
}

export function whatsappAudioDownloadUrl(downloads?: VariantLike): string {
  return audioDownloadVariants(downloads).whatsapp || "";
}
