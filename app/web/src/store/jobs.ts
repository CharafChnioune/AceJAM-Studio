import { create } from "zustand";
import { persist } from "zustand/middleware";

export type JobKind =
  | "generation"
  | "song-batch"
  | "album"
  | "mflux"
  | "mlx-video"
  | "lora"
  | "lora-benchmark"
  | "lora-sweep"
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
  deletePath?: string;
}

interface JobsSlice {
  jobs: Record<string, JobEntry>;
  selectedJobId: string | null;
  dismissedJobIds: Record<string, number>;
  addJob: (entry: JobEntry) => void;
  updateJob: (id: string, patch: Partial<JobEntry>) => void;
  removeJob: (id: string) => void;
  dismissJob: (id: string) => void;
  restoreDismissedJob: (id: string) => void;
  resetJobs: () => void;
  openJob: (id: string) => void;
  closeJob: () => void;
}

export const useJobsStore = create<JobsSlice>()(
  persist(
    (set) => ({
      jobs: {},
      selectedJobId: null,
      dismissedJobIds: {},
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
      dismissJob: (id) =>
        set((s) => {
          const next = { ...s.jobs };
          delete next[id];
          return {
            jobs: next,
            selectedJobId: s.selectedJobId === id ? null : s.selectedJobId,
            dismissedJobIds: { ...s.dismissedJobIds, [id]: Date.now() },
          };
        }),
      restoreDismissedJob: (id) =>
        set((s) => {
          if (!s.dismissedJobIds[id]) return s;
          const nextDismissed = { ...s.dismissedJobIds };
          delete nextDismissed[id];
          return { dismissedJobIds: nextDismissed };
        }),
      resetJobs: () => set({ jobs: {}, selectedJobId: null }),
      openJob: (id) => set({ selectedJobId: id }),
      closeJob: () => set({ selectedJobId: null }),
    }),
    {
      name: "acejam-job-tracker-v1",
      partialize: (state) => ({ dismissedJobIds: state.dismissedJobIds }),
    },
  ),
);
