import * as React from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Sparkles, Wand2, AlertTriangle, ExternalLink } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  chatModelDetails,
  getLLMCatalog,
  promptAssistantRun,
  PROVIDER_LABEL,
  type LLMProvider,
  type WizardMode,
} from "@/lib/api";
import { useSettingsStore } from "@/store/settings";
import { useWizardStore } from "@/store/wizard";
import { toast } from "@/components/ui/sonner";

interface AIPromptStepProps {
  mode: WizardMode;
  placeholder: string;
  examples?: string[];
  onHydrated?: (payload: Record<string, unknown>) => void;
}

const EMPTY_WARNINGS: string[] = [];
const EMPTY_PASTE: Array<{ label?: string; content: string }> = [];

function modelKey(provider: LLMProvider, name: string): string {
  return `${provider}:${name}`;
}

export function AIPromptStep({ mode, placeholder, examples, onHydrated }: AIPromptStepProps) {
  const prompt = useWizardStore((s) => s.prompts[mode]) ?? "";
  const warnings = useWizardStore((s) => s.warnings[mode]) ?? EMPTY_WARNINGS;
  const pasteBlocks = useWizardStore((s) => s.pasteBlocks[mode]) ?? EMPTY_PASTE;
  const setPrompt = useWizardStore((s) => s.setPrompt);
  const setHydration = useWizardStore((s) => s.setHydration);

  const plannerProvider = useSettingsStore((s) => s.plannerProvider);
  const plannerModel = useSettingsStore((s) => s.plannerModel);
  const setPlanner = useSettingsStore((s) => s.setPlanner);

  const catalogQuery = useQuery({
    queryKey: ["llm-catalog"],
    queryFn: getLLMCatalog,
    staleTime: 30_000,
  });

  const allChatModels = React.useMemo(
    () => chatModelDetails(catalogQuery.data),
    [catalogQuery.data],
  );

  // Group by provider for the dropdown
  const grouped = React.useMemo(() => {
    const map = new Map<LLMProvider, typeof allChatModels>();
    for (const m of allChatModels) {
      const arr = map.get(m.provider) ?? [];
      arr.push(m);
      map.set(m.provider, arr);
    }
    return map;
  }, [allChatModels]);

  // Auto-pick a sensible default planner from catalog.settings or first chat model
  React.useEffect(() => {
    if (plannerModel || !catalogQuery.data) return;
    const settings = catalogQuery.data.settings;
    const preferred = settings?.chat_model;
    const preferredProvider = (settings?.provider ?? "ollama") as LLMProvider;
    if (preferred && allChatModels.some((m) => m.name === preferred)) {
      setPlanner(preferredProvider, preferred);
      return;
    }
    if (allChatModels.length > 0) {
      const first = allChatModels[0];
      setPlanner(first.provider, first.name);
    }
  }, [catalogQuery.data, allChatModels, plannerModel, setPlanner]);

  const currentKey = plannerModel ? modelKey(plannerProvider, plannerModel) : "";

  const aiFill = useMutation({
    mutationFn: () =>
      promptAssistantRun({
        mode,
        user_prompt: prompt,
        planner_lm_provider: plannerProvider,
        planner_model: plannerModel || undefined,
      }),
    onSuccess: (data) => {
      if (!data.success) {
        toast.error(data.error || "AI-fill mislukte");
        return;
      }
      setHydration(mode, {
        payload: data.payload,
        warnings: data.warnings,
        paste_blocks: data.paste_blocks,
      });
      if (data.payload) onHydrated?.(data.payload);
      toast.success("AI heeft het wizard-formulier voorgevuld.");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const onModelChange = (key: string) => {
    if (!key) return;
    const idx = key.indexOf(":");
    if (idx < 0) return;
    const provider = key.slice(0, idx) as LLMProvider;
    const name = key.slice(idx + 1);
    setPlanner(provider, name);
  };

  const isLoading = catalogQuery.isLoading;
  const isEmpty = !isLoading && allChatModels.length === 0;

  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <Label htmlFor={`${mode}-prompt`} className="text-sm font-medium">
          Wat wil je maken?
        </Label>
        <Textarea
          id={`${mode}-prompt`}
          placeholder={placeholder}
          value={prompt}
          onChange={(e) => setPrompt(mode, e.target.value)}
          rows={6}
          className="text-base leading-relaxed"
        />
        {examples && examples.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            <span className="text-xs text-muted-foreground">Voorbeelden:</span>
            {examples.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => setPrompt(mode, ex)}
                className="rounded-full border border-border/60 px-2.5 py-0.5 text-xs text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground"
              >
                {ex}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="grid gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
        <div className="space-y-2">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">
            Planner model
          </Label>
          {isEmpty ? (
            <div className="rounded-md border border-yellow-500/30 bg-yellow-500/5 p-3 text-xs text-yellow-200">
              Geen chat-modellen gevonden bij Ollama, LM Studio of de officiële
              ACE-Step LM. Open <Link to="/settings" className="underline">Settings</Link> of
              install een Ollama model met <code className="rounded bg-background/40 px-1">ollama pull qwen3:14b</code>.
            </div>
          ) : (
            <Select value={currentKey} onValueChange={onModelChange}>
              <SelectTrigger>
                <SelectValue placeholder={isLoading ? "Catalog laden…" : "Kies een planner-model"} />
              </SelectTrigger>
              <SelectContent>
                {Array.from(grouped.entries()).map(([provider, list]) => (
                  <SelectGroup key={provider}>
                    <SelectLabel>{PROVIDER_LABEL[provider]}</SelectLabel>
                    {list.map((m) => {
                      const dropdownLabel =
                        m.profile?.dropdown_label || m.display_name || m.name;
                      return (
                        <SelectItem key={m.key} value={modelKey(m.provider, m.name)}>
                          <div className="flex items-center gap-2">
                            <span>{dropdownLabel}</span>
                            {m.size_gb && (
                              <span className="text-[10px] text-muted-foreground">
                                {m.size_gb.toFixed(1)} GB
                              </span>
                            )}
                          </div>
                        </SelectItem>
                      );
                    })}
                  </SelectGroup>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
        <Button
          size="lg"
          onClick={() => aiFill.mutate()}
          disabled={!prompt.trim() || !plannerModel || aiFill.isPending}
          className="gap-2"
        >
          <Sparkles className="size-4" />
          {aiFill.isPending ? "AI denkt na…" : "Vul met AI"}
        </Button>
      </div>

      {(warnings.length > 0 || pasteBlocks.length > 0) && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-3 rounded-xl border border-border/60 bg-card/40 p-4"
        >
          {warnings.length > 0 && (
            <div className="flex gap-2 text-sm">
              <AlertTriangle className="size-4 shrink-0 text-yellow-400" />
              <ul className="space-y-1">
                {warnings.map((w, i) => (
                  <li key={i} className="text-yellow-200/90">{w}</li>
                ))}
              </ul>
            </div>
          )}
          {pasteBlocks.length > 0 && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">
                <Wand2 className="size-3" /> Paste blocks
              </div>
              {pasteBlocks.map((b, i) => (
                <details key={i} className="rounded-md border border-border/40 bg-background/40 p-2 text-xs">
                  <summary className="cursor-pointer font-medium">
                    {b.label || `Block ${i + 1}`}
                  </summary>
                  <pre className="mt-2 whitespace-pre-wrap font-mono text-[11px] text-muted-foreground">
                    {b.content}
                  </pre>
                </details>
              ))}
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
