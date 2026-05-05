import { create } from "zustand";

export type JobKind =
  | "generation"
  | "album"
  | "mflux"
  | "mlx-video"
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
  state?: string;
  stage?: string;
  kindLabel?: string;
  detailsPath?: string;
  logPath?: string;
  updatedAt?: number | string;
  metadata?: Record<string, unknown>;
  error?: string;
  startedAt: number;
}

interface JobsSlice {
  jobs: Record<string, JobEntry>;
  selectedJobId: string | null;
  addJob: (entry: JobEntry) => void;
  updateJob: (id: string, patch: Partial<JobEntry>) => void;
  removeJob: (id: string) => void;
  openJob: (id: string) => void;
  closeJob: () => void;
}

export const useJobsStore = create<JobsSlice>((set) => ({
  jobs: {},
  selectedJobId: null,
  addJob: (entry) =>
    set((s) => ({
      jobs: {
        ...s.jobs,
        [entry.id]: {
          ...s.jobs[entry.id],
          ...entry,
          startedAt: s.jobs[entry.id]?.startedAt || entry.startedAt || Date.now(),
        },
      },
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
      return { jobs: next, selectedJobId: s.selectedJobId === id ? null : s.selectedJobId };
    }),
  openJob: (id) => set({ selectedJobId: id }),
  closeJob: () => set({ selectedJobId: null }),
}));
