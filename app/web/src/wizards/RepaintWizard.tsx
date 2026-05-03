import { SourceAudioWizard } from "./SourceAudioWizard";

export function RepaintWizard() {
  return (
    <SourceAudioWizard
      config={{
        mode: "repaint",
        variant: "repaint",
        title: "Repaint wizard",
        subtitle:
          "Vervang een sectie van een bestaande track terwijl het skelet behouden blijft.",
        defaultModel: "acestep-v15-xl-base",
        examples: [
          "vervang de tweede chorus met een gestripte versie, alleen piano",
          "schilder de bridge over met meer drums en energie",
          "verander de outro naar een ambient fade",
        ],
      }}
    />
  );
}
