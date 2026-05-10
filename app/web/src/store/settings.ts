import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { LLMProvider } from "@/lib/api";

interface SettingsSlice {
  plannerProvider: LLMProvider;
  plannerModel: string;
  setPlanner: (provider: LLMProvider, model: string) => void;
  embeddingProvider: LLMProvider;
  embeddingModel: string;
  setEmbedding: (provider: LLMProvider, model: string) => void;
}

export const useSettingsStore = create<SettingsSlice>()(
  persist(
    (set) => ({
      plannerProvider: "ollama",
      plannerModel: "",
      setPlanner: (provider, model) =>
        set({ plannerProvider: provider, plannerModel: model }),
      embeddingProvider: "ollama",
      embeddingModel: "",
      setEmbedding: (provider, model) =>
        set({ embeddingProvider: provider, embeddingModel: model }),
    }),
    { name: "acejam-settings" },
  ),
);
