import { create } from "zustand";

export type JobKind =
  | "album"
  | "lora"
  | "ollama-pull"
  | "model-download"
  | "lm-download";

export interface JobEntry {
  id: string;
  kind: JobKind;
  label: string;
  progress?: number;
  status?: string;
  startedAt: number;
}

interface JobsSlice {
  jobs: Record<string, JobEntry>;
  addJob: (entry: JobEntry) => void;
  updateJob: (id: string, patch: Partial<JobEntry>) => void;
  removeJob: (id: string) => void;
}

export const useJobsStore = create<JobsSlice>((set) => ({
  jobs: {},
  addJob: (entry) =>
    set((s) => ({
      jobs: { ...s.jobs, [entry.id]: { ...entry, startedAt: entry.startedAt || Date.now() } },
    })),
  updateJob: (id, patch) =>
    set((s) =>
      s.jobs[id]
        ? { jobs: { ...s.jobs, [id]: { ...s.jobs[id], ...patch } } }
        : { jobs: s.jobs },
    ),
  removeJob: (id) =>
    set((s) => {
      const next = { ...s.jobs };
      delete next[id];
      return { jobs: next };
    }),
}));
