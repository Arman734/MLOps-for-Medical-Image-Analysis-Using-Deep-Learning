import { useEffect, useRef, useState } from "react";
import { api, timeAgo, type Job, type RunDetail, type RunSummary, type SystemInfo } from "../api";
import { AlertIcon, CheckIcon, ChevronIcon, DownloadIcon, PlayIcon, RefreshIcon, StopIcon } from "./Icons";
import LineChart, { type Series } from "./LineChart";

type Props = {
  system: SystemInfo | null;
  onAfterTraining: () => Promise<void> | void;
  notify: (message: string, tone?: "ok" | "error") => void;
};

const CONFIGS = [
  { value: "configs/isic_unet.yaml", label: "ISIC 2018 · U-Net segmentation" },
  { value: "configs/pneumonia_densenet.yaml", label: "PneumoniaMNIST · DenseNet-121" },
];

function duration(seconds: number | null | undefined): string {
  if (!seconds || Number.isNaN(seconds)) return "—";
  const total = Math.round(seconds);
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return `${minutes}m ${rest.toString().padStart(2, "0")}s`;
}

function primaryMetric(run: RunSummary): { label: string; value: number } | null {
  const order: [string, string][] = [
    ["val_dice", "dice"],
    ["test_dice", "dice"],
    ["val_auc", "auc"],
    ["test_auc", "auc"],
    ["test_accuracy", "acc"],
  ];
  for (const [key, label] of order) {
    const value = run.metrics[key];
    if (typeof value === "number" && Number.isFinite(value)) return { label, value };
  }
  return null;
}

export default function ExperimentsPanel({ system, onAfterTraining, notify }: Props) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState<RunDetail | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [config, setConfig] = useState(CONFIGS[0].value);
  const [epochs, setEpochs] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const logRef = useRef<HTMLPreElement>(null);

  const loadRuns = async () => {
    setLoading(true);
    try {
      const data = await api.experiments(40);
      setRuns(data.runs);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load experiments");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadRuns();
    api
      .jobs()
      .then(({ jobs }) => {
        const latest = [...jobs].sort((a, b) => (b.started_at ?? 0) - (a.started_at ?? 0))[0];
        if (latest) {
          setJob({ ...latest, log: [], log_offset: 0 });
          api.job(latest.id).then((full) => setJob((prev) => (prev && prev.id === full.id ? { ...full, log_offset: full.log?.length ?? 0 } : prev)));
        }
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    let alive = true;
    const timer = setInterval(async () => {
      try {
        const fresh = await api.job(job.id, job.log_offset ?? 0);
        if (!alive) return;
        setJob((prev) => {
          if (!prev || prev.id !== fresh.id) return fresh;
          return {
            ...fresh,
            log: [...(prev.log ?? []), ...(fresh.log ?? [])].slice(-600),
            log_offset: (fresh.log_offset ?? 0) + (fresh.log?.length ?? 0),
          };
        });
        if (fresh.status !== "running") {
          await loadRuns();
          await onAfterTraining();
          notify(`Training job ${fresh.status}`, fresh.status === "succeeded" ? "ok" : "error");
        }
      } catch {
        /* transient polling error */
      }
    }, 1800);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [job?.log?.length]);

  const startTraining = async () => {
    setError(null);
    try {
      const overrides = epochs.trim() ? [`training.epochs=${Number(epochs)}`] : [];
      const started = await api.startTraining(config, overrides);
      setJob({ ...started, log: [], log_offset: 0 });
      notify("Training job queued — streaming logs below");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start training");
    }
  };

  const cancel = async () => {
    if (!job) return;
    try {
      setJob(await api.cancelJob(job.id));
      notify("Training cancelled");
    } catch (err) {
      notify(err instanceof Error ? err.message : "Cancel failed", "error");
    }
  };

  const openRun = async (runId: string) => {
    try {
      setSelected(await api.run(runId));
    } catch (err) {
      notify(err instanceof Error ? err.message : "Could not load run", "error");
    }
  };

  const lossSeries: Series[] = selected
    ? [
        ...(selected.history.train_loss
          ? [{ label: "train loss", color: "#22d3ee", points: selected.history.train_loss.map((p) => ({ x: p.step, y: p.value })) }]
          : []),
        ...(selected.history.val_loss
          ? [{ label: "val loss", color: "#a78bfa", points: selected.history.val_loss.map((p) => ({ x: p.step, y: p.value })) }]
          : []),
      ]
    : [];
  const metricSeries: Series[] = selected
    ? [
        ...(selected.history.val_dice
          ? [{ label: "dice", color: "#a3e635", points: selected.history.val_dice.map((p) => ({ x: p.step, y: p.value })) }]
          : []),
        ...(selected.history.val_auc
          ? [{ label: "auc", color: "#a3e635", points: selected.history.val_auc.map((p) => ({ x: p.step, y: p.value })) }]
          : []),
      ]
    : [];

  return (
    <section className="stack">
      <div className="section-head">
        <div>
          <span className="eyebrow">MLflow experiments</span>
          <h2>Runs &amp; retraining pipeline</h2>
          <p className="muted">
            Tracked at <span className="mono">{system?.tracking?.tracking_uri ?? "sqlite:///mlflow.db"}</span>. Launch a
            new run below — it reuses the same DVC-prepared data and registers a model version on completion.
          </p>
        </div>
        <div className="head-actions">
          {system?.tracking?.ui_url && (
            <a className="btn btn-ghost" href={system.tracking.ui_url} target="_blank" rel="noreferrer">
              MLflow UI
            </a>
          )}
          <button type="button" className="btn btn-ghost" onClick={() => void loadRuns()} disabled={loading}>
            <RefreshIcon size={15} /> {loading ? "Loading…" : "Refresh"}
          </button>
        </div>
      </div>

      {error && (
        <div className="banner error">
          <AlertIcon size={16} />
          <span>{error}</span>
        </div>
      )}

      <div className="panel">
        <div className="panel-head">
          <div>
            <span className="eyebrow">Pipeline trigger</span>
            <h2>New training run</h2>
          </div>
          <span className="chip mono">local GPU</span>
        </div>
        <div className="train-form">
          <label className="field">
            <span>Configuration</span>
            <select value={config} onChange={(event) => setConfig(event.target.value)} disabled={job?.status === "running"}>
              {CONFIGS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field narrow">
            <span>Epochs override</span>
            <input
              type="number"
              min={1}
              max={200}
              placeholder="from config"
              value={epochs}
              onChange={(event) => setEpochs(event.target.value)}
              disabled={job?.status === "running"}
            />
          </label>
          <button type="button" className="btn btn-primary" onClick={startTraining} disabled={job?.status === "running"}>
            <PlayIcon size={15} /> Start training
          </button>
          {job?.status === "running" && (
            <button type="button" className="btn btn-ghost" onClick={cancel}>
              <StopIcon size={14} /> Cancel
            </button>
          )}
        </div>
        <p className="fineprint">
          Runs execute <span className="mono">medmlops train</span> as a subprocess, log parameters/metrics/artifacts to
          MLflow, write a checkpoint to <span className="mono">artifacts/models</span> and refresh the API automatically.
        </p>
      </div>

      {job && (
        <div className="panel console-panel">
          <div className="panel-head">
            <div>
              <span className="eyebrow">Job {job.id}</span>
              <h2>Training console</h2>
            </div>
            <div className="head-actions">
              <span className={`badge ${job.status === "running" ? "info" : job.status === "succeeded" ? "ok" : "alert"}`}>
                {job.status}
              </span>
              <span className="mono small muted">{job.config.split("/").pop()}</span>
            </div>
          </div>
          <pre className="console" ref={logRef}>
            {(job.log ?? []).join("\n") || "waiting for output…"}
          </pre>
        </div>
      )}

      <div className="panel">
        <div className="panel-head">
          <div>
            <span className="eyebrow">History</span>
            <h2>Experiment runs</h2>
          </div>
          <span className="chip mono">{runs.length} runs</span>
        </div>

        {runs.length ? (
          <div className="table-wrap">
            <table className="table clickable">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Status</th>
                  <th>Metric</th>
                  <th>Epochs</th>
                  <th>Duration</th>
                  <th>Commit</th>
                  <th>When</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const metric = primaryMetric(run);
                  return (
                    <tr
                      key={run.run_id}
                      onClick={() => void openRun(run.run_id)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          void openRun(run.run_id);
                        }
                      }}
                      tabIndex={0}
                      className={selected?.run_id === run.run_id ? "active" : ""}
                    >
                      <td>
                        <div className="run-name">{run.run_name ?? run.run_id.slice(0, 8)}</div>
                        <div className="mono muted tiny">{run.tags?.task ?? run.params?.task ?? "—"}</div>
                      </td>
                      <td>
                        <span className={`badge ${run.status === "FINISHED" ? "ok" : run.status === "RUNNING" ? "info" : run.status === "FAILED" ? "alert" : "warn"}`}>
                          {run.status?.toLowerCase()}
                        </span>
                      </td>
                      <td className="mono">
                        {metric ? (
                          <>
                            {metric.value.toFixed(4)} <span className="muted tiny">{metric.label}</span>
                          </>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className="mono">{run.params["training.epochs"] ?? "—"}</td>
                      <td className="mono">{duration(run.duration_s)}</td>
                      <td className="mono muted">{run.tags?.git_commit?.slice(0, 7) ?? "—"}</td>
                      <td className="muted small">{timeAgo(run.start_time)}</td>
                      <td className="right">
                        <ChevronIcon size={15} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">
            {loading ? "Loading…" : "No runs yet. Start the first training job above — it takes a few minutes on GPU."}
          </p>
        )}
      </div>

      {selected && (
        <div className="panel run-detail">
          <div className="panel-head">
            <div>
              <span className="eyebrow">Run detail</span>
              <h2>{selected.run_name ?? selected.run_id.slice(0, 8)}</h2>
            </div>
            <div className="head-actions">
              <span className={`badge ${selected.status === "FINISHED" ? "ok" : "info"}`}>{selected.status?.toLowerCase()}</span>
              <a className="btn btn-ghost xs" href={`${system?.tracking?.ui_url ?? "http://127.0.0.1:5000"}/#/experiments/${selected.experiment_id}/runs/${selected.run_id}`} target="_blank" rel="noreferrer">
                Open
              </a>
              <button type="button" className="btn btn-ghost xs" onClick={() => setSelected(null)}>
                Close
              </button>
            </div>
          </div>

          <div className="detail-grid">
            <div>
              <h3 className="chart-title">Loss</h3>
              <LineChart series={lossSeries} height={200} yFormat={(v) => v.toFixed(3)} />
            </div>
            <div>
              <h3 className="chart-title">Validation metric</h3>
              <LineChart series={metricSeries} height={200} yFormat={(v) => v.toFixed(3)} />
            </div>
          </div>

          <div className="detail-grid">
            <div>
              <h3 className="chart-title">Metrics</h3>
              <div className="kv-grid mono">
                {Object.entries(selected.metrics)
                  .filter(([, value]) => Number.isFinite(value))
                  .sort((a, b) => a[0].localeCompare(b[0]))
                  .map(([key, value]) => (
                    <div key={key} className="kv-cell">
                      <span className="muted">{key}</span>
                      <strong>{Number(value).toFixed(4)}</strong>
                    </div>
                  ))}
              </div>
            </div>
            <div>
              <h3 className="chart-title">Parameters</h3>
              <div className="kv-grid mono">
                {Object.entries(selected.params)
                  .sort((a, b) => a[0].localeCompare(b[0]))
                  .map(([key, value]) => (
                    <div key={key} className="kv-cell">
                      <span className="muted">{key}</span>
                      <strong className="truncate" title={value}>{value}</strong>
                    </div>
                  ))}
              </div>
            </div>
          </div>

          <div className="tag-row">
            {Object.entries(selected.tags)
              .filter(([key]) => !key.startsWith("mlflow."))
              .map(([key, value]) => (
                <span className="tag" key={key}>
                  <span className="muted">{key}</span> {value}
                </span>
              ))}
          </div>
        </div>
      )}

      <div className="split-grid">
        <div className="panel">
          <div className="panel-head">
            <div>
              <span className="eyebrow">Reproducibility</span>
              <h2>Data lineage</h2>
            </div>
            <DownloadIcon size={16} />
          </div>
          <ul className="kv-list">
            <li>
              <span>Version control</span>
              <span className="mono">DVC pipeline · dvc.yaml</span>
            </li>
            <li>
              <span>ISIC segmentation</span>
              <span className="mono">{system?.datasets?.segmentation?.splits?.train?.count ?? "—"} train / {system?.datasets?.segmentation?.splits?.test?.count ?? "—"} test</span>
            </li>
            <li>
              <span>PneumoniaMNIST</span>
              <span className="mono">{system?.datasets?.classification?.splits?.train?.count ?? "—"} train / {system?.datasets?.classification?.splits?.test?.count ?? "—"} test</span>
            </li>
            <li>
              <span>DagsHub switch</span>
              <span className="mono">medmlops train --dagshub</span>
            </li>
          </ul>
        </div>
        <div className="panel">
          <div className="panel-head">
            <div>
              <span className="eyebrow">CI/CD</span>
              <h2>Automation</h2>
            </div>
            <CheckIcon size={16} />
          </div>
          <ul className="kv-list">
            <li>
              <span>Unit + API tests</span>
              <span className="mono">pytest · GitHub Actions</span>
            </li>
            <li>
              <span>Image build</span>
              <span className="mono">Docker · GHCR</span>
            </li>
            <li>
              <span>Pipeline replay</span>
              <span className="mono">dvc repro</span>
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
}
