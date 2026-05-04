import * as React from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface AppErrorBoundaryState {
  error: Error | null;
}

export class AppErrorBoundary extends React.Component<
  React.PropsWithChildren,
  AppErrorBoundaryState
> {
  state: AppErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("[AceJAM UI]", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <div className="flex min-h-screen items-center justify-center bg-background p-6 text-foreground">
        <div className="w-full max-w-xl rounded-lg border bg-card p-6 shadow-xl">
          <div className="mb-4 flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-md bg-destructive/15 text-destructive">
              <AlertTriangle className="size-5" />
            </div>
            <div>
              <h1 className="text-lg font-semibold">AceJAM UI crashte</h1>
              <p className="text-sm text-muted-foreground">
                De app blijft draaien. Herlaad de interface om verder te gaan.
              </p>
            </div>
          </div>
          <pre className="mb-4 max-h-40 overflow-auto rounded-md bg-background/70 p-3 text-xs text-muted-foreground">
            {this.state.error.message}
          </pre>
          <Button onClick={() => window.location.reload()} className="gap-2">
            <RefreshCw className="size-4" />
            Herlaad UI
          </Button>
        </div>
      </div>
    );
  }
}
