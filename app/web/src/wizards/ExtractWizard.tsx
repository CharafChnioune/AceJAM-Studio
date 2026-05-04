import { SourceAudioWizard } from "./SourceAudioWizard";

export function ExtractWizard() {
  return (
    <SourceAudioWizard
      config={{
        mode: "extract",
        variant: "extract",
        title: "Extract wizard",
        subtitle:
          "Isoleer specifieke stems (vocals, drums, bass…) uit een track.",
        defaultModel: "acestep-v15-xl-sft",
        examples: [
          "haal alleen de vocals eruit",
          "isoleer drums en bass voor een instrumentale mix",
          "extract guitar en keyboard stems",
        ],
      }}
    />
  );
}
