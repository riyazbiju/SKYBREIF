import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const aviationButtonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap font-medium transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        primary:
          "bg-primary text-primary-foreground hover:bg-primary/90 cockpit-glow rounded-lg",
        secondary:
          "bg-secondary text-secondary-foreground border border-border hover:bg-secondary/80 hover:border-primary/50 rounded-lg",
        ghost:
          "text-foreground hover:bg-secondary/50 rounded-lg",
        action:
          "aviation-gradient text-primary-foreground font-semibold shadow-lg hover:shadow-xl hover:scale-105 rounded-xl",
        outline:
          "border-2 border-primary text-primary bg-transparent hover:bg-primary hover:text-primary-foreground rounded-lg",
      },
      size: {
        sm: "h-9 px-4 text-sm",
        md: "h-11 px-6 text-base",
        lg: "h-14 px-8 text-lg",
        xl: "h-16 px-10 text-xl",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "md",
    },
  }
);

export interface AviationButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof aviationButtonVariants> {}

const AviationButton = React.forwardRef<HTMLButtonElement, AviationButtonProps>(
  ({ className, variant, size, ...props }, ref) => {
    return (
      <button
        className={cn(aviationButtonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
AviationButton.displayName = "AviationButton";

export { AviationButton, aviationButtonVariants };
