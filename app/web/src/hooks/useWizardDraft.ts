import * as React from "react";
import type { FieldValues, Path, UseFormReturn } from "react-hook-form";
import { useWizardStore } from "@/store/wizard";

function jsonableRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object") return {};
  return JSON.parse(JSON.stringify(value)) as Record<string, unknown>;
}

function normalizeAceStepRenderDraft<T extends Record<string, unknown>>(values: T): T {
  const songModel = String(values.song_model ?? "");
  if (!songModel.startsWith("acestep-v15-")) return values;
  const normalized: Record<string, unknown> = { ...values };
  if (songModel.includes("turbo")) {
    if ("inference_steps" in normalized && Number(normalized.inference_steps) < 8) {
      normalized.inference_steps = 8;
    }
    if ("shift" in normalized) normalized.shift = 3;
    return normalized as T;
  }
  if ("inference_steps" in normalized && Number(normalized.inference_steps) < 50) {
    normalized.inference_steps = 50;
  }
  if ("shift" in normalized) normalized.shift = 1;
  return normalized as T;
}

export function mergeWizardDraft<T extends object>(
  defaults: T,
  draft?: Record<string, unknown>,
): T {
  return normalizeAceStepRenderDraft({ ...defaults, ...(draft ?? {}) } as Record<string, unknown>) as T;
}

export function useWizardDraft<T extends FieldValues>(
  mode: string,
  form: UseFormReturn<T, unknown, T>,
) {
  const setDraft = useWizardStore((s) => s.setDraft);
  const clearDraft = useWizardStore((s) => s.clearDraft);
  const timerRef = React.useRef<number | null>(null);

  const saveNow = React.useCallback(
    (values?: T | Record<string, unknown>) => {
      if (timerRef.current) {
        window.clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      setDraft(mode, normalizeAceStepRenderDraft(jsonableRecord(values ?? form.getValues())));
    },
    [form, mode, setDraft],
  );

  const clear = React.useCallback(() => {
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    clearDraft(mode);
  }, [clearDraft, mode]);

  React.useEffect(() => {
    const subscription = form.watch((values) => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
      timerRef.current = window.setTimeout(() => {
        setDraft(mode, normalizeAceStepRenderDraft(jsonableRecord(values)));
      }, 250);
    });
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
      subscription.unsubscribe();
    };
  }, [form, mode, setDraft]);

  return { saveNow, clear };
}

export function usePromptMirror<T extends FieldValues>(
  form: UseFormReturn<T, unknown, T>,
  fieldName: keyof T & string,
  prompt: string | undefined,
) {
  const previousPromptRef = React.useRef(prompt ?? "");

  React.useEffect(() => {
    const nextPrompt = prompt ?? "";
    const previousPrompt = previousPromptRef.current;
    previousPromptRef.current = nextPrompt;
    if (!nextPrompt) return;
    const fieldPath = fieldName as Path<T>;
    const current = String(form.getValues(fieldPath) ?? "");
    const isInitialRestoredDraft =
      previousPrompt === nextPrompt && current.trim().length > 0 && current !== nextPrompt;
    if (isInitialRestoredDraft) return;
    if (current !== nextPrompt) {
      form.setValue(fieldPath, nextPrompt as never, {
        shouldValidate: true,
        shouldDirty: true,
      });
    }
  }, [fieldName, form, prompt]);
}
