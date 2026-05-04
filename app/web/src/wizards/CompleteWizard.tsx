import { SourceAudioWizard } from "./SourceAudioWizard";

export function CompleteWizard() {
  return (
    <SourceAudioWizard
      config={{
        mode: "complete",
        variant: "complete",
        title: "Complete wizard",
        subtitle:
          "Vul ontbrekende stems aan op een bestaande arrangement.",
        defaultModel: "acestep-v15-xl-sft",
        examples: [
          "voeg drums toe aan deze instrumentale piano-track",
          "completeer het arrangement met bass en strings",
        ],
      }}
    />
  );
}
