import * as React from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Sparkles, Disc3, Music2, ArrowRight } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const STORAGE_KEY = "acejam-welcome-seen";

const QUICK_PATHS = [
  {
    to: "/wizard/simple",
    title: "Eén track snel",
    blurb: "Beschrijf een idee, AI vult de rest, ACE-Step rendert.",
    icon: Sparkles,
  },
  {
    to: "/wizard/album",
    title: "Volledig album",
    blurb: "Concept → tracklist met cover en per-track artwork.",
    icon: Disc3,
  },
  {
    to: "/library",
    title: "Bekijk library",
    blurb: "Alle eerder gegenereerde tracks afspelen of remixen.",
    icon: Music2,
  },
];

export function WelcomeDialog() {
  const [open, setOpen] = React.useState(false);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    if (!window.localStorage.getItem(STORAGE_KEY)) {
      setOpen(true);
    }
  }, []);

  const dismiss = () => {
    window.localStorage.setItem(STORAGE_KEY, String(Date.now()));
    setOpen(false);
  };

  return (
    <Dialog open={open} onOpenChange={(v) => (!v ? dismiss() : setOpen(true))}>
      <DialogContent className="max-w-xl gap-6">
        <div className="space-y-2">
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-[10px] font-medium uppercase tracking-widest text-primary"
          >
            <Sparkles className="size-3" /> Welkom
          </motion.div>
          <DialogTitle className="font-display text-3xl font-semibold leading-tight">
            Maak muziek met één <span className="text-primary">prompt</span>.
          </DialogTitle>
          <DialogDescription className="text-balance">
            Elke wizard start met je idee in eigen woorden. De AI vult elk veld
            alvast in — jij loopt langs en stuurt bij waar je wilt.
          </DialogDescription>
        </div>

        <div className="grid gap-2">
          {QUICK_PATHS.map((p, i) => {
            const Icon = p.icon;
            return (
              <motion.div
                key={p.to}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 + i * 0.05 }}
              >
                <Link
                  to={p.to}
                  onClick={dismiss}
                  className={cn(
                    "group flex items-center gap-3 rounded-xl border bg-card/50 p-3 transition-all",
                    "hover:-translate-y-0.5 hover:border-primary/40 hover:bg-card hover:shadow-md",
                  )}
                >
                  <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
                    <Icon className="size-4" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="font-display text-sm font-semibold">{p.title}</p>
                    <p className="truncate text-xs text-muted-foreground">{p.blurb}</p>
                  </div>
                  <ArrowRight className="size-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-0.5 group-hover:text-foreground" />
                </Link>
              </motion.div>
            );
          })}
        </div>

        <div className="flex items-center justify-between gap-2 border-t border-border/60 pt-4 text-xs text-muted-foreground">
          <span>Tip: open Settings om je AI- en image-modellen te kiezen.</span>
          <Button variant="ghost" size="sm" onClick={dismiss}>
            Begin
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
