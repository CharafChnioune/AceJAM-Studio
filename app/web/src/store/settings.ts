import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { LLMProvider } from "@/lib/api";

interface SettingsSlice {
  plannerProvider: LLMProvider;
  plannerModel: string;
  artProvider: LLMProvider;
  artModel: string;
  setPlanner: (provider: LLMProvider, model: string) => void;
  setArt: (provider: LLMProvider, model: string) => void;
}

export const useSettingsStore = create<SettingsSlice>()(
  persist(
    (set) => ({
      plannerProvider: "ollama",
      plannerModel: "",
      artProvider: "ollama",
      artModel: "",
      setPlanner: (provider, model) =>
        set({ plannerProvider: provider, plannerModel: model }),
      setArt: (provider, model) =>
        set({ artProvider: provider, artModel: model }),
    }),
    { name: "acejam-settings" },
  ),
);
