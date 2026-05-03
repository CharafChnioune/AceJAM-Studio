import * as React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowLeft, ArrowRight, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

export interface WizardStepDef {
  key: string;
  title: string;
  description?: string;
  /** When false, Next button is disabled. */
  isValid?: boolean;
  /** Optional hidden flag — used to skip steps based on state. */
  hidden?: boolean;
  render: () => React.ReactNode;
}

interface WizardShellProps {
  title: string;
  subtitle?: string;
  steps: WizardStepDef[];
  step: number;
  onStepChange: (next: number) => void;
  onFinish?: () => void;
  finishLabel?: string;
  isFinishing?: boolean;
  className?: string;
}

const variants = {
  enter: (dir: number) => ({ x: 24 * dir, opacity: 0 }),
  center: { x: 0, opacity: 1 },
  exit: (dir: number) => ({ x: -24 * dir, opacity: 0 }),
};

export function WizardShell({
  title,
  subtitle,
  steps,
  step,
  onStepChange,
  onFinish,
  finishLabel = "Genereer",
  isFinishing = false,
  className,
}: WizardShellProps) {
  const visibleSteps = steps.filter((s) => !s.hidden);
  const safeStep = Math.min(Math.max(step, 0), visibleSteps.length - 1);
  const current = visibleSteps[safeStep];
  const direction = React.useRef(1);

  const goPrev = () => {
    if (safeStep <= 0) return;
    direction.current = -1;
    onStepChange(safeStep - 1);
  };
  const goNext = () => {
    if (current.isValid === false) return;
    if (safeStep >= visibleSteps.length - 1) {
      onFinish?.();
      return;
    }
    direction.current = 1;
    onStepChange(safeStep + 1);
  };

  const progress = Math.round(((safeStep + 1) / visibleSteps.length) * 100);
  const isLast = safeStep >= visibleSteps.length - 1;

  return (
    <div className={cn("flex h-full min-h-0 flex-col", className)}>
      <header className="flex flex-col gap-3 border-b border-border/60 px-6 py-5 sm:px-10">
        <div className="flex items-baseline justify-between gap-4">
          <div className="space-y-1">
            <p className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
              Wizard
            </p>
            <h1 className="font-display text-2xl font-semibold text-balance sm:text-3xl">
              {title}
            </h1>
            {subtitle && (
              <p className="max-w-2xl text-sm text-muted-foreground">{subtitle}</p>
            )}
          </div>
          <div className="hidden text-right sm:block">
            <p className="text-xs uppercase tracking-widest text-muted-foreground">
              Stap {safeStep + 1} / {visibleSteps.length}
            </p>
            <p className="text-base font-medium">{current.title}</p>
          </div>
        </div>
        <Progress value={progress} className="h-1" />
        <nav className="hidden flex-wrap gap-1.5 sm:flex">
          {visibleSteps.map((s, idx) => {
            const reached = idx <= safeStep;
            const active = idx === safeStep;
            return (
              <button
                key={s.key}
                type="button"
                onClick={() => idx <= safeStep && onStepChange(idx)}
                className={cn(
                  "group flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                  active &&
                    "border-primary/40 bg-primary/10 text-primary-foreground",
                  !active && reached && "border-border/80 text-foreground/80",
                  !reached && "border-border/40 text-muted-foreground/60",
                )}
                disabled={idx > safeStep}
              >
                <span
                  className={cn(
                    "flex size-5 items-center justify-center rounded-full text-[10px] font-semibold",
                    active && "bg-primary text-primary-foreground",
                    !active && reached && "bg-secondary",
                    !reached && "bg-muted",
                  )}
                >
                  {idx < safeStep ? <Check className="size-3" /> : idx + 1}
                </span>
                <span className="hidden md:inline">{s.title}</span>
              </button>
            );
          })}
        </nav>
      </header>

      <main className="relative flex-1 overflow-y-auto px-6 py-8 sm:px-10 scroll-fade-y">
        <AnimatePresence mode="wait" custom={direction.current}>
          <motion.section
            key={current.key}
            custom={direction.current}
            variants={variants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
            className="mx-auto w-full max-w-3xl space-y-6"
          >
            <div className="space-y-1">
              <h2 className="font-display text-xl font-semibold">{current.title}</h2>
              {current.description && (
                <p className="text-sm text-muted-foreground">{current.description}</p>
              )}
            </div>
            {current.render()}
          </motion.section>
        </AnimatePresence>
      </main>

      <footer className="flex items-center justify-between border-t border-border/60 px-6 py-4 sm:px-10">
        <Button
          variant="ghost"
          onClick={goPrev}
          disabled={safeStep === 0}
          className="text-muted-foreground"
        >
          <ArrowLeft className="size-4" />
          Vorige
        </Button>
        <div className="flex items-center gap-3">
          <span className="hidden text-xs text-muted-foreground sm:inline">
            {progress}%
          </span>
          <Button
            onClick={goNext}
            disabled={current.isValid === false || isFinishing}
            size="lg"
          >
            {isLast ? finishLabel : "Volgende"}
            {!isLast && <ArrowRight className="size-4" />}
          </Button>
        </div>
      </footer>
    </div>
  );
}

export interface FieldGroupProps {
  title?: string;
  description?: string;
  children: React.ReactNode;
  className?: string;
}

export function FieldGroup({ title, description, children, className }: FieldGroupProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className={cn("space-y-3 rounded-xl border bg-card/40 p-5", className)}
    >
      {(title || description) && (
        <div className="space-y-0.5">
          {title && <h3 className="text-sm font-semibold">{title}</h3>}
          {description && (
            <p className="text-xs text-muted-foreground">{description}</p>
          )}
        </div>
      )}
      {children}
    </motion.div>
  );
}
