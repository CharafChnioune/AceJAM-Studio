import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { LLMProvider } from "@/lib/api";

interface SettingsSlice {
  plannerProvider: LLMProvider;
  plannerModel: string;
  setPlanner: (provider: LLMProvider, model: string) => void;
}

export const useSettingsStore = create<SettingsSlice>()(
  persist(
    (set) => ({
      plannerProvider: "ollama",
      plannerModel: "",
      setPlanner: (provider, model) =>
        set({ plannerProvider: provider, plannerModel: model }),
    }),
    { name: "acejam-settings" },
  ),
);
