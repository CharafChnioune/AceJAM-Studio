import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SettingsSlice {
  plannerProvider: string;
  plannerModel: string;
  artModel: string;
  setPlanner: (provider: string, model: string) => void;
  setArtModel: (model: string) => void;
}

export const useSettingsStore = create<SettingsSlice>()(
  persist(
    (set) => ({
      plannerProvider: "ollama",
      plannerModel: "",
      artModel: "",
      setPlanner: (provider, model) =>
        set({ plannerProvider: provider, plannerModel: model }),
      setArtModel: (model) => set({ artModel: model }),
    }),
    { name: "acejam-settings" },
  ),
);
