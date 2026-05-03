import { SourceAudioWizard } from "./SourceAudioWizard";

export function CoverWizard() {
  return (
    <SourceAudioWizard
      config={{
        mode: "cover",
        variant: "cover",
        title: "Cover wizard",
        subtitle:
          "Upload een track en hercomponeer hem in een nieuwe stijl met dezelfde structuur.",
        defaultModel: "acestep-v15-xl-sft",
        examples: [
          "maak een akoestische versie van deze track, alleen piano + vocals",
          "remix tot een uptempo dance-track op 128 bpm",
          "vertaal naar lo-fi hip-hop, voeg vinyl-crackle toe",
        ],
      }}
    />
  );
}
