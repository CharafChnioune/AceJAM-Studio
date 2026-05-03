import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, Disc3, GraduationCap, Download, X, Brain } from "lucide-react";
import { useJobsStore, type JobKind } from "@/store/jobs";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

const ICONS: Record<JobKind, React.ComponentType<{ className?: string }>> = {
  album: Disc3,
  lora: GraduationCap,
  "ollama-pull": Brain,
  "model-download": Download,
  "lm-download": Download,
};

export function JobTracker() {
  const jobs = useJobsStore((s) => s.jobs);
  const removeJob = useJobsStore((s) => s.removeJob);
  const list = Object.values(jobs).sort((a, b) => a.startedAt - b.startedAt);

  if (list.length === 0) return null;

  return (
    <div className="space-y-1.5 px-2 pb-2">
      <p className="px-1 text-[10px] uppercase tracking-widest text-muted-foreground">
        Achtergrond-jobs
      </p>
      <AnimatePresence initial={false}>
        {list.map((job) => {
          const Icon = ICONS[job.kind] ?? Loader2;
          const done = job.status === "complete" || job.status === "error";
          return (
            <motion.div
              key={job.id}
              layout
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: -8 }}
              transition={{ duration: 0.16 }}
              className={cn(
                "group flex items-center gap-2 rounded-md border bg-card/50 px-2 py-1.5 text-xs",
                job.status === "error" && "border-destructive/40",
                job.status === "complete" && "border-primary/40",
              )}
            >
              {done ? (
                <Icon className="size-3.5 shrink-0 text-primary" />
              ) : (
                <Loader2 className="size-3.5 shrink-0 animate-spin text-primary" />
              )}
              <div className="min-w-0 flex-1 space-y-1">
                <p className="truncate font-medium">{job.label}</p>
                {typeof job.progress === "number" && !done && (
                  <Progress value={job.progress} className="h-0.5" />
                )}
                {job.status && (
                  <p className="truncate text-[10px] text-muted-foreground">{job.status}</p>
                )}
              </div>
              {done && (
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="size-5 opacity-0 transition-opacity group-hover:opacity-100"
                  onClick={() => removeJob(job.id)}
                  aria-label="Sluit"
                >
                  <X className="size-3" />
                </Button>
              )}
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
