import { NavLink, Route, Routes } from "react-router-dom";
import {
  Music2,
  ListMusic,
  Image as ImageIcon,
  Video,
  Disc3,
  GraduationCap,
  Library as LibraryIcon,
  Settings as SettingsIcon,
  Home as HomeIcon,
} from "lucide-react";
import { Home } from "@/pages/Home";
import { Library } from "@/pages/Library";
import { Settings } from "@/pages/Settings";
import { SimpleWizard } from "@/wizards/SimpleWizard";
import { CustomWizard } from "@/wizards/CustomWizard";
import { BatchSongsWizard } from "@/wizards/BatchSongsWizard";
import { AlbumWizard } from "@/wizards/AlbumWizard";
import { CoverWizard } from "@/wizards/CoverWizard";
import { RepaintWizard } from "@/wizards/RepaintWizard";
import { ExtractWizard } from "@/wizards/ExtractWizard";
import { LegoWizard } from "@/wizards/LegoWizard";
import { CompleteWizard } from "@/wizards/CompleteWizard";
import { NewsWizard } from "@/wizards/NewsWizard";
import { TrainerWizard } from "@/wizards/TrainerWizard";
import { ImageWizard } from "@/wizards/ImageWizard";
import { ImageTrainerWizard } from "@/wizards/ImageTrainerWizard";
import { VideoWizard } from "@/wizards/VideoWizard";
import { Button } from "@/components/ui/button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { JobTracker } from "@/components/JobTracker";
import { WelcomeDialog } from "@/components/WelcomeDialog";
import { cn } from "@/lib/utils";

function Sidebar() {
  const items = [
    { to: "/", label: "Home", icon: HomeIcon },
    { to: "/wizard/simple", label: "Music", icon: Music2 },
    { to: "/wizard/batch", label: "Batch", icon: ListMusic },
    { to: "/wizard/image", label: "Images", icon: ImageIcon },
    { to: "/wizard/video", label: "Video", icon: Video },
    { to: "/wizard/album", label: "Albums", icon: Disc3 },
    { to: "/wizard/trainer", label: "Training", icon: GraduationCap },
    { to: "/library", label: "Library", icon: LibraryIcon },
    { to: "/settings", label: "Settings", icon: SettingsIcon },
  ];
  return (
    <aside className="flex w-16 shrink-0 flex-col items-center gap-3 border-r border-border/60 bg-sidebar/80 py-5 backdrop-blur sm:w-56 sm:items-stretch sm:px-3">
      <div className="flex items-center justify-center gap-2 sm:justify-start sm:px-2">
        <div className="flex size-9 items-center justify-center rounded-xl bg-primary/15 text-primary">
          <Music2 className="size-4" />
        </div>
        <span className="hidden font-display text-base font-semibold sm:inline">
          MLX Media
        </span>
      </div>
      <nav className="mt-2 flex flex-col gap-1">
        {items.map((it) => {
          const Icon = it.icon;
          return (
            <NavLink
              key={it.to}
              to={it.to}
              end={it.to === "/"}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-2 rounded-md px-2 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary/15 text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                )
              }
              title={it.label}
            >
              <Icon className="size-4 shrink-0" />
              <span className="hidden sm:inline">{it.label}</span>
            </NavLink>
          );
        })}
      </nav>
      <div className="mt-auto hidden flex-col gap-2 px-2 text-[11px] text-muted-foreground sm:flex">
        <JobTracker />
      </div>
    </aside>
  );
}

export default function App() {
  return (
    <TooltipProvider delayDuration={200}>
      <WelcomeDialog />
      <div className="flex h-full min-h-0 w-full">
        <Sidebar />
        <div className="flex min-w-0 flex-1 flex-col">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/wizard/simple" element={<SimpleWizard />} />
            <Route path="/wizard/batch" element={<BatchSongsWizard />} />
            <Route path="/wizard/custom" element={<CustomWizard />} />
            <Route path="/wizard/album" element={<AlbumWizard />} />
            <Route path="/wizard/cover" element={<CoverWizard />} />
            <Route path="/wizard/repaint" element={<RepaintWizard />} />
            <Route path="/wizard/extract" element={<ExtractWizard />} />
            <Route path="/wizard/lego" element={<LegoWizard />} />
            <Route path="/wizard/complete" element={<CompleteWizard />} />
            <Route path="/wizard/news" element={<NewsWizard />} />
            <Route path="/wizard/trainer" element={<TrainerWizard />} />
            <Route path="/wizard/image" element={<ImageWizard />} />
            <Route path="/wizard/image-trainer" element={<ImageTrainerWizard />} />
            <Route path="/wizard/video" element={<VideoWizard />} />
            <Route path="/library" element={<Library />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Home />} />
          </Routes>
        </div>
      </div>
      <div className="fixed inset-x-3 bottom-3 z-40 sm:hidden">
        <JobTracker compact />
      </div>
    </TooltipProvider>
  );
}
