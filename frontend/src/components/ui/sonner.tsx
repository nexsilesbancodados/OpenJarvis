import { Toaster as Sonner, type ToasterProps } from "sonner"
import { CircleCheckIcon, InfoIcon, TriangleAlertIcon, OctagonXIcon, Loader2Icon } from "lucide-react"

const Toaster = ({ ...props }: ToasterProps) => {
  return (
    <Sonner
      className="toaster group"
      icons={{
        success: (
          <CircleCheckIcon className="size-4" />
        ),
        info: (
          <InfoIcon className="size-4" />
        ),
        warning: (
          <TriangleAlertIcon className="size-4" />
        ),
        error: (
          <OctagonXIcon className="size-4" />
        ),
        loading: (
          <Loader2Icon className="size-4 animate-spin" />
        ),
      }}
      // The nova tokens first, with the shadcn ones as fallback. The rebuilt
      // index.css defines neither --popover nor --border, so referencing them
      // alone rendered every toast with undefined colours.
      style={
        {
          "--normal-bg": "var(--nova-panel-solid, var(--popover, #101420))",
          "--normal-text": "var(--nova-text, var(--popover-foreground, #f5f7ff))",
          "--normal-border": "var(--nova-border, var(--border, rgba(255,255,255,.14)))",
          "--border-radius": "var(--radius, 12px)",
        } as React.CSSProperties
      }
      toastOptions={{
        classNames: {
          toast: "cn-toast",
        },
      }}
      {...props}
    />
  )
}

export { Toaster }
