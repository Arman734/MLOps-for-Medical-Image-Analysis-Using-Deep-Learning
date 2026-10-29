import { useCallback, useEffect, useState } from "react";
import { api, formatMs, formatPct, timeAgo, type MonitoringSummary, type StorageStats } from "../api";
import { ActivityIcon, AlertIcon, CheckIcon, DatabaseIcon, DownloadIcon, RefreshIcon, ShieldIcon, XIcon } from "./Icons";
import StatCard from "./StatCard";

type Props = {
  notify: (message: string, tone?: "ok" | "error") => void;
  refreshSignal: number;
};

const STATUS_TONE: Record<string, string> = {
  stable: "ok",
  watch: "warn",
  alert: "alert",
  insufficient_data: "subtle",
  no_reference: "subtle",
};

export default function MonitoringPanel({ notify, refreshSignal }: Props) {
  const [summary, setSummary] = useState<MonitoringSummary | null>(null);
  const [storage, setStorage] = useState<StorageStats | null>(null);
  const [task, setTask] = useState<"segmentation" | "classification">("segmentation");
  const [taskChosen, setTaskChosen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [next, stats] = await Promise.all([api.monitoring(), api.storage()]);
      setSummary(next);
      setStorage(stats);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Monitoring unavailable");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), 15000);
    return () => clearInterval(timer);
  }, [load, refreshSignal]);

  const feedback = async (predictionId: string, verdict: "correct" | "incorrect") => {
    try {
      await api.feedback(predictionId, verdict);
      notify("Feedback recorded");
      await load();
    } catch (err) {
      notify(err instanceof Error ? err.message : "Feedback failed", "error");
    }
  };

  const prune = async () => {
    if (!window.confirm("Delete predictions older than 30 days? This cannot be undone.")) return;
    try {
      const result = await api.prune(30);
      notify(`Pruned ${result.removed_predictions} predictions and ${result.removed_feedback} feedback rows`);
      await load();
    } catch (err) {
      notify(err instanceof Error ? err.message : "Prune failed", "error");
    }
  };

  useEffect(() => {
    if (taskChosen || !summary) return;
    const hasReference = (name: "segmentation" | "classification") => {
      const entry = summary.drift?.[name];
      return Boolean(entry && entry.status !== "no_reference");
    };
    const servedSeg = summary.by_task?.segmentation?.predictions ?? 0;
    const servedCls = summary.by_task?.classification?.predictions ?? 0;
    if (hasReference("classification") && (!hasReference("segmentation") || servedCls > servedSeg)) {
      setTask("classification");
    }
  }, [summary, taskChosen]);

  const chooseTask = (next: "segmentation" | "classification") => {
    setTaskChosen(true);
    setTask(next);
  };

  const servedTasks = Object.values(summary?.by_task ?? {}).filter((entry) => entry.predictions > 0);
  const servedCount = servedTasks.reduce((acc, entry) => acc + entry.predictions, 0);
  const weightedLatency = servedCount
    ? servedTasks.reduce((acc, entry) => acc + entry.avg_latency_ms * entry.predictions, 0) / servedCount
    : null;

  const drift = summary?.drift?.[task];
  const status = drift?.status ?? "no_reference";
  const labels = summary?.labels?.[task] ?? {};
  const labelTotal = Object.values(labels).reduce((acc, value) => acc + value, 0) || 1;
  const feedbackTotal = Object.values(summary?.feedback ?? {}).reduce((acc, value) => acc + value, 0);
  const agree = summary?.feedback?.correct ?? 0;

  return (
    <section className="stack">
      <div className="section-head">
        <div>
          <span className="eyebrow">Production telemetry</span>
          <h2>Monitoring &amp; drift</h2>
          <p className="muted">
            Every served prediction is logged with its image statistics. Distribution shift is tracked with
            population-stability index (PSI) against the training reference.
          </p>
        </div>
        <div className="head-actions">
          <a className="btn btn-ghost" href="/api/monitoring/export.csv">
            <DownloadIcon size={15} /> Export CSV
          </a>
          <button type="button" className="btn btn-ghost" onClick={() => void load()} disabled={loading}>
            <RefreshIcon size={15} /> {loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {error && (
        <div className="banner error">
          <AlertIcon size={16} />
          <span>{error}</span>
        </div>
      )}

      <div className="stat-grid four">
        <StatCard
          label="Predictions served"
          value={summary?.total_predictions ?? "—"}
          sub={`${summary?.by_task?.segmentation?.predictions ?? 0} segmentation · ${summary?.by_task?.classification?.predictions ?? 0} triage`}
          icon={<ActivityIcon size={16} />}
          tone="accent"
        />
        <StatCard
          label="P95 latency"
          value={formatMs(summary?.recent_latency_p95_ms)}
          sub={weightedLatency !== null ? `avg ${formatMs(weightedLatency)}` : "no data yet"}
          icon={<ShieldIcon size={16} />}
        />
        <StatCard
          label="Clinician agreement"
          value={feedbackTotal ? formatPct(agree / feedbackTotal, 0) : "—"}
          sub={`${feedbackTotal} reviewed · ${summary?.feedback?.incorrect ?? 0} flagged`}
          tone={feedbackTotal && agree / feedbackTotal < 0.8 ? "warn" : "ok"}
          icon={<CheckIcon size={16} />}
        />
        <StatCard
          label="Input drift"
          value={status.replace(/_/g, " ")}
          sub={
            status === "no_reference"
              ? "baseline missing"
              : status === "insufficient_data"
                ? `${drift?.window ?? 0}/${drift?.min_samples ?? 150} predictions`
                : `${drift?.window ?? 0} predictions in window`
          }
          tone={status === "alert" ? "alert" : status === "watch" ? "warn" : status === "stable" ? "ok" : "default"}
          icon={<AlertIcon size={16} />}
        />
      </div>

      <div className="split-grid wide-left">
        <div className="panel">
          <div className="panel-head">
            <div>
              <span className="eyebrow">Covariate shift</span>
              <h2>Drift report</h2>
            </div>
            <div className="segmented small" role="tablist">
              <button
                type="button"
                className={task === "segmentation" ? "active" : ""}
                onClick={() => chooseTask("segmentation")}
              >
                segmentation
              </button>
              <button
                type="button"
                className={task === "classification" ? "active" : ""}
                onClick={() => chooseTask("classification")}
              >
                classification
              </button>
            </div>
          </div>

          {drift?.status === "insufficient_data" ? (
            <div>
              <p className="muted">
                Collecting live {task} traffic — {drift.window}/{drift.min_samples ?? 150} predictions needed before
                PSI scores become statistically meaningful.
              </p>
              <div className="prob-track" style={{ marginTop: 14 }}>
                <div
                  className="prob-fill active"
                  style={{
                    width: `${Math.min(100, (drift.window / (drift.min_samples ?? 150)) * 100)}%`,
                  }}
                />
              </div>
            </div>
          ) : drift && Object.keys(drift.features).length ? (
            <div className="drift-list">
              {Object.entries(drift.features).map(([name, feature]) => {
                const width = Math.min(100, (feature.psi / 0.4) * 100);
                return (
                  <div className="drift-row" key={name}>
                    <div className="drift-label">
                      <span>{name}</span>
                      <span className="mono muted tiny">
                        ref {feature.reference_mean.toFixed(3)} → now {feature.current_mean.toFixed(3)}
                      </span>
                    </div>
                    <div className="drift-track">
                      <span className="threshold t-watch" />
                      <span className="threshold t-alert" />
                      <div className={`drift-fill ${feature.status}`} style={{ width: `${Math.max(width, 1.5)}%` }} />
                    </div>
                    <span className="mono drift-psi">{feature.psi.toFixed(3)}</span>
                    <span className={`badge ${STATUS_TONE[feature.status] ?? "subtle"}`}>{feature.status}</span>
                  </div>
                );
              })}
              <p className="fineprint">
                PSI thresholds: &lt; 0.10 stable · 0.10–0.25 watch · &gt; 0.25 alert. Reference histograms were captured
                from the training split during data preparation.
              </p>
            </div>
          ) : (
            <p className="muted">
              No drift baseline yet for {task}. Train the model and serve some predictions — the baseline is written by
              the data-prep stage.
            </p>
          )}
        </div>

        <div className="panel">
          <div className="panel-head">
            <div>
              <span className="eyebrow">Serving mix</span>
              <h2>Label distribution</h2>
            </div>
          </div>
          {Object.keys(labels).length ? (
            <div className="prob-bars">
              {Object.entries(labels).map(([label, count]) => (
                <div className="prob-row" key={label}>
                  <span className="prob-label">{label.replace(/_/g, " ")}</span>
                  <div className="prob-track">
                    <div className="prob-fill active" style={{ width: `${Math.max((count / labelTotal) * 100, 2)}%` }} />
                  </div>
                  <span className="prob-value mono">{count}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">No predictions for {task} yet.</p>
          )}
          <div className="divider" />
          <div className="mono small muted">
            {task === "segmentation"
              ? "Segmentation labels: lesion vs no_lesion (area > 0.5%)."
              : "Classification labels come from the model's argmax class."}
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <div>
            <span className="eyebrow">Audit trail</span>
            <h2>Recent predictions</h2>
          </div>
          <span className="chip mono">{summary?.total_predictions ?? 0} logged</span>
        </div>
        {summary?.recent?.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>When</th>
                  <th>Task</th>
                  <th>File</th>
                  <th>Model</th>
                  <th>Label</th>
                  <th>Confidence</th>
                  <th>Latency</th>
                  <th>Feedback</th>
                </tr>
              </thead>
              <tbody>
                {summary.recent.map((row) => (
                  <tr key={row.id}>
                    <td className="muted small">{timeAgo(row.ts)}</td>
                    <td>
                      <span className={`badge ${row.task === "segmentation" ? "info" : "violet"}`}>{row.task.slice(0, 4)}</span>
                    </td>
                    <td className="mono small truncate" title={row.filename ?? ""}>
                      {row.filename ?? "—"}
                    </td>
                    <td className="mono small">{row.model_id}</td>
                    <td className="mono small">{row.label ?? "—"}</td>
                    <td className="mono">{row.confidence !== null ? formatPct(row.confidence) : "—"}</td>
                    <td className="mono">{formatMs(row.latency_ms)}</td>
                    <td>
                      {row.feedback ? (
                        <span className={`badge ${row.feedback.verdict === "correct" ? "ok" : row.feedback.verdict === "incorrect" ? "alert" : "warn"}`}>
                          {row.feedback.verdict}
                        </span>
                      ) : (
                        <div className="feedback-actions">
                          <button type="button" className="icon-btn ok" aria-label="Mark correct" onClick={() => void feedback(row.id, "correct")}>
                            <CheckIcon size={13} />
                          </button>
                          <button type="button" className="icon-btn alert" aria-label="Mark incorrect" onClick={() => void feedback(row.id, "incorrect")}>
                            <XIcon size={13} />
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">No predictions logged yet. Run a case from the Analyze tab.</p>
        )}
      </div>

      <div className="panel">
        <div className="panel-head">
          <div>
            <span className="eyebrow">Persistence</span>
            <h2>Ledger storage</h2>
          </div>
          <span className={`badge ${storage?.wal ? "ok" : "warn"}`}>
            <DatabaseIcon size={12} /> {storage?.wal ? "WAL" : storage?.journal_mode ?? "unknown"}
          </span>
        </div>
        {storage ? (
          <>
            <ul className="kv-list">
              <li>
                <span>Database</span>
                <span className="mono small">{storage.path}</span>
              </li>
              <li>
                <span>Size</span>
                <span className="mono">{(storage.size_bytes / 1024).toFixed(1)} KiB</span>
              </li>
              <li>
                <span>Rows</span>
                <span className="mono">
                  {storage.predictions} predictions · {storage.feedback} feedback
                </span>
              </li>
              <li>
                <span>Oldest entry</span>
                <span className="mono">{storage.oldest_ts ? timeAgo(storage.oldest_ts) : "—"}</span>
              </li>
            </ul>
            <div className="head-actions" style={{ marginTop: 14 }}>
              <a className="btn btn-ghost xs" href={`/api/monitoring/export.csv?task=${task}`}>
                <DownloadIcon size={13} /> Export {task}
              </a>
              <button type="button" className="btn btn-ghost xs" onClick={() => void prune()}>
                Prune older than 30 days
              </button>
            </div>
          </>
        ) : (
          <p className="muted">Ledger status unavailable.</p>
        )}
      </div>
    </section>
  );
}
