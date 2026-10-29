type IconProps = { size?: number; className?: string };

const strokeProps = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.7,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

export function PulseIcon({ size = 20, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="M3 12h4l2.5-6 4 12 2.5-6H21" {...strokeProps} />
    </svg>
  );
}

export function ScanIcon({ size = 20, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="M4 8V5a1 1 0 0 1 1-1h3M20 8V5a1 1 0 0 0-1-1h-3M4 16v3a1 1 0 0 0 1 1h3M20 16v3a1 1 0 0 1-1 1h-3" {...strokeProps} />
      <circle cx="12" cy="12" r="3" {...strokeProps} />
      <path d="M12 3v2M12 19v2M3 12h2M19 12h2" {...strokeProps} />
    </svg>
  );
}

export function UploadIcon({ size = 20, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="M12 16V4m0 0 4.5 4.5M12 4 7.5 8.5" {...strokeProps} />
      <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" {...strokeProps} />
    </svg>
  );
}

export function BrainIcon({ size = 20, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="M9.5 3.5A3 3 0 0 0 6.6 7 3.2 3.2 0 0 0 5 10c0 1 .4 1.9 1.1 2.5A3.4 3.4 0 0 0 6 15.4 3.3 3.3 0 0 0 9.5 20 2.4 2.4 0 0 0 12 18.4V5.6a2.4 2.4 0 0 0-2.5-2.1Z" {...strokeProps} />
      <path d="M14.5 3.5A3 3 0 0 1 17.4 7 3.2 3.2 0 0 1 19 10c0 1-.4 1.9-1.1 2.5A3.4 3.4 0 0 0 18 15.4 3.3 3.3 0 0 1 14.5 20 2.4 2.4 0 0 1 12 18.4" {...strokeProps} />
      <path d="M9 9.5h2M13 13h2M9 15h2" {...strokeProps} />
    </svg>
  );
}

export function FlaskIcon({ size = 20, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="M9 3h6M10 3v5.2L5.4 17a2.4 2.4 0 0 0 2.1 3.5h9a2.4 2.4 0 0 0 2.1-3.5L14 8.2V3" {...strokeProps} />
      <path d="M7.5 14h9" {...strokeProps} />
    </svg>
  );
}

export function ActivityIcon({ size = 20, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <circle cx="12" cy="12" r="9" {...strokeProps} />
      <path d="M7 12h2.5l1.5-3.5 2.5 7 1.5-3.5H17" {...strokeProps} />
    </svg>
  );
}

export function LayersIcon({ size = 20, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="m12 3 9 5-9 5-9-5 9-5Z" {...strokeProps} />
      <path d="m3 13 9 5 9-5M3 17l9 5 9-5" {...strokeProps} />
    </svg>
  );
}

export function CpuIcon({ size = 20, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <rect x="6" y="6" width="12" height="12" rx="2" {...strokeProps} />
      <rect x="9.5" y="9.5" width="5" height="5" rx="1" {...strokeProps} />
      <path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3" {...strokeProps} />
    </svg>
  );
}

export function DatabaseIcon({ size = 20, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <ellipse cx="12" cy="6" rx="7.5" ry="3" {...strokeProps} />
      <path d="M4.5 6v6c0 1.66 3.36 3 7.5 3s7.5-1.34 7.5-3V6" {...strokeProps} />
      <path d="M4.5 12v6c0 1.66 3.36 3 7.5 3s7.5-1.34 7.5-3v-6" {...strokeProps} />
    </svg>
  );
}

export function CheckIcon({ size = 16, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="m5 12.5 4.5 4.5L19 7" {...strokeProps} />
    </svg>
  );
}

export function XIcon({ size = 16, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="m6 6 12 12M18 6 6 18" {...strokeProps} />
    </svg>
  );
}

export function AlertIcon({ size = 18, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="M12 4 3 20h18L12 4Z" {...strokeProps} />
      <path d="M12 10v4M12 17h.01" {...strokeProps} />
    </svg>
  );
}

export function RefreshIcon({ size = 16, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="M20 11a8 8 0 1 0-2.3 5.7" {...strokeProps} />
      <path d="M20 5v6h-6" {...strokeProps} />
    </svg>
  );
}

export function DownloadIcon({ size = 16, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="M12 4v12m0 0 4.5-4.5M12 16l-4.5-4.5" {...strokeProps} />
      <path d="M4 18v2h16v-2" {...strokeProps} />
    </svg>
  );
}

export function PlayIcon({ size = 16, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="M7 4.5v15l12-7.5-12-7.5Z" {...strokeProps} />
    </svg>
  );
}

export function StopIcon({ size = 16, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <rect x="6" y="6" width="12" height="12" rx="2" {...strokeProps} />
    </svg>
  );
}

export function ChevronIcon({ size = 16, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="m9 6 6 6-6 6" {...strokeProps} />
    </svg>
  );
}

export function CrosshairIcon({ size = 20, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <circle cx="12" cy="12" r="8" {...strokeProps} />
      <path d="M12 2v4M12 18v4M2 12h4M18 12h4" {...strokeProps} />
    </svg>
  );
}

export function ShieldIcon({ size = 18, className }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="M12 3 5 6v5c0 4.5 3 8.2 7 10 4-1.8 7-5.5 7-10V6l-7-3Z" {...strokeProps} />
      <path d="m9 12 2 2 4-4" {...strokeProps} />
    </svg>
  );
}
