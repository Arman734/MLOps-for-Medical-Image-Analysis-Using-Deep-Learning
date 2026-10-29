import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Health, type ModelsResponse, type SystemInfo } from "./api";
import Analyzer from "./components/Analyzer";
import ExperimentsPanel from "./components/ExperimentsPanel";
import Header, { type Tab } from "./components/Header";
import ModelsPanel from "./components/ModelsPanel";
import MonitoringPanel from "./components/MonitoringPanel";

type Toast = { id: number; message: string; tone: "ok" | "error" };

const TABS: Tab[] = ["analyze", "models", "experiments", "monitoring"];

function initialTab(): Tab {
  const hash = window.location.hash.replace("#", "") as Tab;
  return TABS.includes(hash) ? hash : "analyze";
}

export default function App() {
  const [tab, setTab] = useState<Tab>(initialTab);
  const [health, setHealth] = useState<Health | null>(null);
  const [reachable, setReachable] = useState(true);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [monitorSignal, setMonitorSignal] = useState(0);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastSeq = useRef(0);
  const syncedReadiness = useRef("");

  const notify = useCallback((message: string, tone: "ok" | "error" = "ok") => {
    const id = (toastSeq.current += 1);
    setToasts((previous) => [...previous.slice(-3), { id, message, tone }]);
    window.setTimeout(() => {
      setToasts((previous) => previous.filter((item) => item.id !== id));
    }, 4200);
  }, []);

  const checkHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
      setReachable(true);
    } catch {
      setHealth(null);
      setReachable(false);
    }
  }, []);

  const loadSystem = useCallback(async () => {
    try {
      setSystem(await api.system());
    } catch {
      /* the health banner already surfaces an unreachable API */
    }
  }, []);

  const loadModels = useCallback(async () => {
    try {
      setModels(await api.models());
    } catch {
      /* the health banner already surfaces an unreachable API */
    }
  }, []);

  useEffect(() => {
    void checkHealth();
    const timer = window.setInterval(() => void checkHealth(), 15000);
    return () => window.clearInterval(timer);
  }, [checkHealth]);

  useEffect(() => {
    void loadModels();
    void loadSystem();
  }, [loadModels, loadSystem]);

  useEffect(() => {
    if (!health || !models) return;
    const key = `${health.segmentation_ready}/${health.classification_ready}`;
    if (syncedReadiness.current === key) return;
    syncedReadiness.current = key;
    if (
      health.segmentation_ready !== models.segmentation_ready ||
      health.classification_ready !== models.classification_ready
    ) {
      void loadModels();
    }
  }, [health, models, loadModels]);

  const refreshModels = useCallback(async () => {
    try {
      setModels(await api.refreshModels());
      await loadSystem();
    } catch (error) {
      notify(error instanceof Error ? error.message : "Could not refresh models", "error");
    }
  }, [notify, loadSystem]);

  const changeTab = useCallback((next: Tab) => {
    setTab(next);
    window.history.replaceState(null, "", `#${next}`);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const onPrediction = useCallback(() => {
    setMonitorSignal((value) => value + 1);
  }, []);

  return (
    <div className="app">
      <Header tab={tab} onTab={changeTab} health={health} />

      <main className="app-main">
        {!reachable && (
          <div className="banner error offline" role="alert">
            <span>
              API unreachable — start the service with <code>make serve</code> (or <code>medmlops serve</code>), then
              retry.
            </span>
            <button type="button" className="btn btn-ghost xs" onClick={() => void checkHealth()}>
              Retry connection
            </button>
          </div>
        )}

        {tab === "analyze" && (
          <Analyzer models={models} notify={notify} onNavigate={changeTab} onPrediction={onPrediction} />
        )}
        {tab === "models" && <ModelsPanel models={models} onRefresh={refreshModels} notify={notify} />}
        {tab === "experiments" && (
          <ExperimentsPanel system={system} onAfterTraining={refreshModels} notify={notify} />
        )}
        {tab === "monitoring" && <MonitoringPanel notify={notify} refreshSignal={monitorSignal} />}
      </main>

      <footer className="footer">
        <span>MedMLOps research prototype · model outputs require clinician review · not a medical device.</span>
        <nav className="footer-links mono" aria-label="Resources">
          <a href="/docs" target="_blank" rel="noreferrer">
            API docs
          </a>
          <a href="/metrics" target="_blank" rel="noreferrer">
            Prometheus
          </a>
          {system?.tracking?.ui_url && (
            <a href={system.tracking.ui_url} target="_blank" rel="noreferrer">
              MLflow
            </a>
          )}
          <span>{system?.device_name ?? health?.device_name ?? "no accelerator"}</span>
          <span>v{health?.version ?? system?.version ?? "1.0"}</span>
        </nav>
      </footer>

      <div className="toast-stack" aria-live="polite" aria-atomic="false">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.tone}`} role="status">
            <span className="toast-dot" />
            {toast.message}
          </div>
        ))}
      </div>
    </div>
  );
}
