import * as React from "react";
import { cn } from "@/lib/utils";

export interface AviationInputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  unit?: string;
}

const AviationInput = React.forwardRef<HTMLInputElement, AviationInputProps>(
  ({ className, type, label, unit, ...props }, ref) => {
    return (
      <div className="space-y-2">
        {label && (
          <label className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
            {label}
          </label>
        )}
        <div className="relative">
          <input
            type={type}
            className={cn(
              "flex h-12 w-full rounded-lg border border-border bg-secondary/50 px-4 py-2 text-foreground font-mono text-base",
              "placeholder:text-muted-foreground/50",
              "focus:outline-none focus:ring-2 focus:ring-primary focus:border-primary",
              "transition-all duration-200",
              "disabled:cursor-not-allowed disabled:opacity-50",
              unit && "pr-12",
              className
            )}
            ref={ref}
            {...props}
          />
          {unit && (
            <span className="absolute right-4 top-1/2 -translate-y-1/2 text-sm text-muted-foreground font-mono">
              {unit}
            </span>
          )}
        </div>
      </div>
    );
  }
);
AviationInput.displayName = "AviationInput";

export { AviationInput };
