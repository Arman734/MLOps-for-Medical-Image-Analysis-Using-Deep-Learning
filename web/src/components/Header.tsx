import type { Health } from "../api";
import { ActivityIcon, BrainIcon, CpuIcon, FlaskIcon, PulseIcon } from "./Icons";

export type Tab = "analyze" | "models" | "experiments" | "monitoring";

const NAV: { id: Tab; label: string; icon: JSX.Element }[] = [
  { id: "analyze", label: "Analyze", icon: <BrainIcon size={17} /> },
  { id: "models", label: "Models", icon: <FlaskIcon size={17} /> },
  { id: "experiments", label: "Experiments", icon: <ActivityIcon size={17} /> },
  { id: "monitoring", label: "Monitoring", icon: <PulseIcon size={17} /> },
];

type Props = {
  tab: Tab;
  onTab: (tab: Tab) => void;
  health: Health | null;
};

export default function Header({ tab, onTab, health }: Props) {
  const online = Boolean(health);
  const device = health?.device === "cuda" ? "CUDA" : health?.device?.toUpperCase() ?? "OFFLINE";
  return (
    <header className="header">
      <div className="brand">
        <div className="brand-mark" aria-hidden>
          <PulseIcon size={22} />
        </div>
        <div className="brand-text">
          <div className="brand-name">
            MED<span>ML</span>OPS
          </div>
          <div className="brand-sub">clinical imaging console</div>
        </div>
      </div>

      <nav className="nav" aria-label="Primary">
        {NAV.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`nav-item ${tab === item.id ? "active" : ""}`}
            onClick={() => onTab(item.id)}
            aria-current={tab === item.id ? "page" : undefined}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="header-status">
        <span className={`status-pill ${online ? "online" : "offline"}`}>
          <span className="status-dot" />
          {online ? "API online" : "API offline"}
        </span>
        <span className="chip mono">
          <CpuIcon size={14} />
          {device}
        </span>
        {health && <span className="chip subtle mono">v{health.version}</span>}
      </div>
    </header>
  );
}
