import { useState } from "react";
import { api, timeAgo, type ModelsResponse, type RegistryModel } from "../api";
import { AlertIcon, CheckIcon, CpuIcon, DatabaseIcon, LayersIcon, RefreshIcon } from "./Icons";

type Props = {
  models: ModelsResponse | null;
  onRefresh: () => Promise<void> | void;
  notify: (message: string, tone?: "ok" | "error") => void;
};

const METRIC_KEYS: Record<string, string[]> = {
  segmentation: ["test_dice", "test_iou", "test_recall", "test_precision"],
  classification: ["test_accuracy", "test_auc", "test_sensitivity", "test_f1_macro"],
};

function championRun(registry: RegistryModel[] | undefined, name: string): string | null {
  const entry = registry?.find((item) => item.name === name);
  if (!entry || !entry.champion) return null;
  const version = entry.versions.find((v) => v.version === entry.champion);
  return version?.run_id ?? null;
}

export default function ModelsPanel({ models, onRefresh, notify }: Props) {
  const [busy, setBusy] = useState<string | null>(null);

  const promote = async (registryName: string, runId: string) => {
    setBusy(runId);
    try {
      const result = await api.promote(registryName, runId);
      notify(`Promoted ${result.name} v${result.version} to champion`);
      await onRefresh();
    } catch (err) {
      notify(err instanceof Error ? err.message : "Promotion failed", "error");
    } finally {
      setBusy(null);
    }
  };

  const sync = async () => {
    setBusy("sync");
    try {
      const result = await api.syncModels();
      if (result.downloaded.length) {
        const suffix = result.errors.length ? ` · ${result.errors.length} unavailable` : "";
        notify(`Synced ${result.downloaded.length} champion checkpoint(s) from MLflow${suffix}`);
      } else {
        const reason = result.errors[0]?.error ?? "registry is empty";
        notify(`No champions to sync — ${reason}`, "error");
      }
      await onRefresh();
    } catch (err) {
      notify(err instanceof Error ? err.message : "Registry sync failed", "error");
    } finally {
      setBusy(null);
    }
  };

  const loaded = models?.models ?? [];
  const registry = models?.registry ?? [];

  return (
    <section className="stack">
      <div className="section-head">
        <div>
          <span className="eyebrow">Model registry</span>
          <h2>Checkpoints &amp; champions</h2>
          <p className="muted">Every training run registers a version in MLflow. The console serves the newest local checkpoint; the champion alias marks the promoted release. Sync pulls champion checkpoints onto a fresh deployment.</p>
        </div>
        <div className="head-actions">
          <button type="button" className="btn btn-ghost" onClick={() => void sync()} disabled={busy !== null}>
            <DatabaseIcon size={15} /> {busy === "sync" ? "Syncing…" : "Sync registry"}
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => void onRefresh()}>
            <RefreshIcon size={15} /> Refresh
          </button>
        </div>
      </div>

      {!loaded.length && (
        <div className="banner warn">
          <AlertIcon size={16} />
          <span>No checkpoints in <span className="mono">{models?.model_dir ?? "artifacts/models"}</span>. Run a training job from Experiments.</span>
        </div>
      )}

      <div className="model-grid">
        {loaded.map((model) => {
          const registryName = registry.find((item) => item.versions.some((v) => v.run_id === model.run_id))?.name;
          const isChampion = registryName ? championRun(registry, registryName) === model.run_id : false;
          const metricKeys = METRIC_KEYS[model.task] ?? [];
          return (
            <article className="model-card" key={model.id}>
              <div className="model-top">
                <div className={`task-glyph ${model.task}`}>
                  <LayersIcon size={20} />
                </div>
                <div className="model-title">
                  <h3>{model.id}</h3>
                  <p className="mono muted small">{model.arch} · {model.dataset}</p>
                </div>
                {isChampion && (
                  <span className="badge champion">
                    <CheckIcon size={12} /> champion
                  </span>
                )}
              </div>

              <div className="model-metrics">
                {metricKeys.map((key) => (
                  <div className="metric-chip" key={key}>
                    <span>{key.replace("test_", "").replace("_", " ")}</span>
                    <strong>{model.metrics[key] !== undefined ? model.metrics[key].toFixed(4) : "—"}</strong>
                  </div>
                ))}
              </div>

              <dl className="model-facts">
                <div>
                  <dt>Task</dt>
                  <dd className="mono">{model.task}</dd>
                </div>
                <div>
                  <dt>Input</dt>
                  <dd className="mono">{model.image_size}²</dd>
                </div>
                <div>
                  <dt>Params</dt>
                  <dd className="mono">{(model.parameters / 1e6).toFixed(2)}M</dd>
                </div>
                <div>
                  <dt>{model.task === "segmentation" ? "Threshold" : "Classes"}</dt>
                  <dd className="mono">
                    {model.task === "segmentation" ? model.threshold.toFixed(2) : model.class_names.join(" / ")}
                  </dd>
                </div>
              </dl>

              <div className="model-foot mono">
                <span>run {model.run_id ? model.run_id.slice(0, 8) : "—"}</span>
                <span>git {model.git_commit ?? "—"}</span>
                <span>{model.created_at ? timeAgo(Date.parse(model.created_at) / 1000) : "—"}</span>
              </div>
            </article>
          );
        })}
      </div>

      <div className="panel">
        <div className="panel-head">
          <div>
            <span className="eyebrow">MLflow model registry</span>
            <h2>Version history</h2>
          </div>
          <span className="chip mono">{registry.length} models</span>
        </div>

        {registry.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Version</th>
                  <th>Run</th>
                  <th>Primary metric</th>
                  <th>Alias</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {registry.flatMap((entry) =>
                  entry.versions.map((version) => {
                    const metrics = version.metrics ?? {};
                    const preferred = ["val_dice", "val_auc", "test_dice", "test_auc", "test_accuracy"];
                    const key = preferred.find((candidate) => metrics[candidate] !== undefined);
                    const isChampion = entry.champion === version.version;
                    return (
                      <tr key={`${entry.name}-${version.version}`}>
                        <td className="mono">{entry.name}</td>
                        <td className="mono">v{version.version}</td>
                        <td className="mono muted">{version.run_id?.slice(0, 8)}</td>
                        <td className="mono">
                          {key ? `${key} ${Number(metrics[key]).toFixed(4)}` : "—"}
                        </td>
                        <td>{isChampion ? <span className="badge champion">champion</span> : <span className="badge subtle">—</span>}</td>
                        <td className="right">
                          <button
                            type="button"
                            className="btn btn-ghost xs"
                            disabled={isChampion || busy === version.run_id}
                            onClick={() => promote(entry.name, version.run_id)}
                          >
                            {busy === version.run_id ? "Promoting…" : "Promote"}
                          </button>
                        </td>
                      </tr>
                    );
                  }),
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">Registry is empty. Finish a training run to create the first version.</p>
        )}
      </div>

      <div className="split-grid">
        <div className="panel">
          <div className="panel-head">
            <div>
              <span className="eyebrow">Serving</span>
              <h2>Runtime</h2>
            </div>
            <CpuIcon size={18} />
          </div>
          <ul className="kv-list">
            <li>
              <span>Device</span>
              <span className="mono">{models?.device_name ?? "—"}</span>
            </li>
            <li>
              <span>Model directory</span>
              <span className="mono small">{models?.model_dir ?? "—"}</span>
            </li>
            <li>
              <span>Segmentation ready</span>
              <span className={`badge ${models?.segmentation_ready ? "ok" : "warn"}`}>{String(Boolean(models?.segmentation_ready))}</span>
            </li>
            <li>
              <span>Classification ready</span>
              <span className={`badge ${models?.classification_ready ? "ok" : "warn"}`}>{String(Boolean(models?.classification_ready))}</span>
            </li>
          </ul>
          {models?.load_errors?.length ? (
            <div className="banner error">
              <AlertIcon size={16} />
              <span>{models.load_errors.map((e) => `${e.file}: ${e.error}`).join(" · ")}</span>
            </div>
          ) : null}
        </div>

        <div className="panel">
          <div className="panel-head">
            <div>
              <span className="eyebrow">Governance</span>
              <h2>Promotion policy</h2>
            </div>
          </div>
          <p className="muted">
            Training automatically promotes a fresh version when its validation metric (Dice for segmentation, AUC for
            classification) beats the current champion. Manual promotion is available above for any registered
            version, enabling rollback by re-promoting an older run.
          </p>
          <ul className="kv-list">
            <li>
              <span>Champion alias</span>
              <span className="mono">champion</span>
            </li>
            <li>
              <span>Signature</span>
              <span className="mono">tensor → tensor</span>
            </li>
            <li>
              <span>Model card</span>
              <span className="mono">logged per run</span>
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
}
