import { SourceAudioWizard } from "./SourceAudioWizard";

export function LegoWizard() {
  return (
    <SourceAudioWizard
      config={{
        mode: "lego",
        variant: "lego",
        title: "Lego wizard",
        subtitle:
          "Reconstrueer een track per stem-laag — herbouw het stuk per puzzelstuk.",
        defaultModel: "acestep-v15-xl-base",
        examples: [
          "bouw vocals + drums + bass terug, voeg pad toe",
          "houd alle stems behalve drums; vervang die met trap",
        ],
      }}
    />
  );
}
