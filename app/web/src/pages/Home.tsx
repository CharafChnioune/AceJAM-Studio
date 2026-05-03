import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Sparkles,
  Music2,
  Disc3,
  Repeat2,
  Brush,
  Scissors,
  Blocks,
  Workflow,
  Newspaper,
  GraduationCap,
  ArrowRight,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface ModeCard {
  to: string;
  title: string;
  blurb: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
  enabled: boolean;
}

const cards: ModeCard[] = [
  {
    to: "/wizard/simple",
    title: "Simple",
    blurb: "Eén prompt → hele track. Snelste pad van idee naar audio.",
    icon: Sparkles,
    badge: "Aanbevolen",
    enabled: true,
  },
  {
    to: "/wizard/custom",
    title: "Custom",
    blurb: "Volledige controle: tags, lyrics, bpm, key, render-instellingen.",
    icon: Music2,
    enabled: true,
  },
  {
    to: "/wizard/album",
    title: "Album",
    blurb: "Meerdere tracks met album-coherentie en cover artwork.",
    icon: Disc3,
    enabled: true,
  },
  {
    to: "/wizard/cover",
    title: "Cover",
    blurb: "Hercomponeer een bestaande audio in nieuwe stijl.",
    icon: Repeat2,
    enabled: true,
  },
  {
    to: "/wizard/repaint",
    title: "Repaint",
    blurb: "Vervang een sectie van een track terwijl het skelet blijft.",
    icon: Brush,
    enabled: true,
  },
  {
    to: "/wizard/extract",
    title: "Extract",
    blurb: "Isoleer vocals, drums, bass of andere stems.",
    icon: Scissors,
    enabled: true,
  },
  {
    to: "/wizard/lego",
    title: "Lego",
    blurb: "Reconstrueer een track per stem-laag.",
    icon: Blocks,
    enabled: true,
  },
  {
    to: "/wizard/complete",
    title: "Complete",
    blurb: "Vul ontbrekende stems aan op een bestaande arrangement.",
    icon: Workflow,
    enabled: true,
  },
  {
    to: "/wizard/news",
    title: "News",
    blurb: "Maak een satirische track op basis van een nieuwsbericht.",
    icon: Newspaper,
    enabled: true,
  },
  {
    to: "/wizard/trainer",
    title: "Trainer",
    blurb: "Train een persoonlijke LoRA op je eigen tracks.",
    icon: GraduationCap,
    enabled: true,
  },
];

export function Home() {
  return (
    <div className="mx-auto w-full max-w-6xl space-y-10 px-6 py-10 sm:px-10 sm:py-14">
      <header className="space-y-3">
        <Badge variant="muted" className="rounded-full px-3 py-1 font-mono text-[10px] uppercase tracking-[0.18em]">
          AceJAM Studio · React UI
        </Badge>
        <h1 className="font-display text-balance text-3xl font-semibold leading-tight sm:text-5xl">
          Welke <span className="text-primary">muziek</span> ga je vandaag maken?
        </h1>
        <p className="max-w-2xl text-balance text-base text-muted-foreground">
          Kies een wizard. Elke flow start met één prompt en de AI vult de rest
          van het formulier alvast in — jij loopt erlangs en stuurt bij.
        </p>
      </header>

      <motion.section
        initial="hidden"
        animate="show"
        variants={{
          hidden: {},
          show: { transition: { staggerChildren: 0.04 } },
        }}
        className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3"
      >
        {cards.map((c) => {
          const Icon = c.icon;
          const className = cn(
            "group relative flex h-full flex-col gap-3 rounded-2xl border bg-card/50 p-5 transition-all",
            c.enabled
              ? "hover:-translate-y-0.5 hover:border-primary/40 hover:bg-card hover:shadow-lg"
              : "cursor-not-allowed opacity-60",
          );
          const Body = (
            <>
                <div className="flex items-center justify-between">
                  <div className="flex size-10 items-center justify-center rounded-xl bg-primary/15 text-primary">
                    <Icon className="size-5" />
                  </div>
                  {c.badge && <Badge variant="outline" className="text-[10px]">{c.badge}</Badge>}
                  {!c.enabled && <Badge variant="muted" className="text-[10px]">Binnenkort</Badge>}
                </div>
                <div className="space-y-1">
                  <h2 className="font-display text-lg font-semibold">{c.title}</h2>
                  <p className="text-sm text-muted-foreground">{c.blurb}</p>
                </div>
                {c.enabled && (
                  <span className="mt-auto flex items-center gap-1.5 text-sm font-medium text-primary opacity-0 transition-opacity group-hover:opacity-100">
                    Start wizard <ArrowRight className="size-3.5" />
                  </span>
                )}
            </>
          );
          return (
            <motion.div
              key={c.to}
              variants={{
                hidden: { opacity: 0, y: 8 },
                show: { opacity: 1, y: 0 },
              }}
            >
              {c.enabled ? (
                <Link to={c.to} className={className}>{Body}</Link>
              ) : (
                <div className={className}>{Body}</div>
              )}
            </motion.div>
          );
        })}
      </motion.section>
    </div>
  );
}
