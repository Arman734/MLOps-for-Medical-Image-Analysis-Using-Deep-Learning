import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  formatMs,
  formatPct,
  imgSrc,
  type ClfResponse,
  type ModelsResponse,
  type SegResponse,
} from "../api";
import { AlertIcon, BrainIcon, CheckIcon, CrosshairIcon, RefreshIcon, ScanIcon, ShieldIcon, XIcon } from "./Icons";
import ImageDropzone from "./ImageDropzone";
import StatCard from "./StatCard";
import Viewer from "./Viewer";

type Mode = "segmentation" | "classification";

type Props = {
  models: ModelsResponse | null;
  onNavigate: (tab: "experiments") => void;
  onPrediction: () => void;
  notify: (message: string, tone?: "ok" | "error") => void;
};

function dataUrlToBlob(dataUrl: string): Blob {
  const [meta, payload] = dataUrl.includes(",") ? dataUrl.split(",") : ["data:image/png;base64", dataUrl];
  const mime = /:(.*?);/.exec(meta)?.[1] ?? "image/png";
  const binary = atob(payload);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

export default function Analyzer({ models, onNavigate, onPrediction, notify }: Props) {
  const [mode, setMode] = useState<Mode>("segmentation");
  const [file, setFile] = useState<File | null>(null);
  const [demo, setDemo] = useState<{ name: string; base64: string } | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const [modelId, setModelId] = useState("");
  const [tta, setTta] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [seg, setSeg] = useState<SegResponse | null>(null);
  const [clf, setClf] = useState<ClfResponse | null>(null);
  const [view, setView] = useState("overlay");
  const [opacity, setOpacity] = useState(0.85);
  const [feedback, setFeedback] = useState<"correct" | "incorrect" | null>(null);
  const objectUrl = useRef<string | null>(null);

  const taskModels = useMemo(
    () => (models?.models ?? []).filter((model) => model.task === mode),
    [models, mode],
  );
  const ready = mode === "segmentation" ? models?.segmentation_ready : models?.classification_ready;

  const previewUrl = useMemo(() => {
    if (file) return URL.createObjectURL(file);
    if (demo) return imgSrc(demo.base64);
    return null;
  }, [file, demo]);

  useEffect(() => {
    if (!previewUrl || !file) return;
    objectUrl.current = previewUrl;
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = null;
    };
  }, [previewUrl, file]);

  useEffect(() => {
    setFile(null);
    setDemo(null);
    setSeg(null);
    setClf(null);
    setModelId("");
    setError(null);
    setFeedback(null);
    setView(mode === "segmentation" ? "overlay" : "gradcam");
  }, [mode]);

  const resetResults = () => {
    setSeg(null);
    setClf(null);
    setFeedback(null);
    setError(null);
  };

  const loadDemo = async () => {
    setDemoLoading(true);
    setError(null);
    try {
      const sample = await api.demo(mode);
      setFile(null);
      setDemo({ name: sample.name, base64: sample.image_base64 });
      setSeg(null);
      setClf(null);
      setFeedback(null);
      notify(`Loaded demo case ${sample.name}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Demo case unavailable");
    } finally {
      setDemoLoading(false);
    }
  };

  const analyze = async () => {
    if (!previewUrl) return;
    setBusy(true);
    setError(null);
    setSeg(null);
    setClf(null);
    setFeedback(null);
    try {
      const blob = file ?? dataUrlToBlob(demo!.base64);
      if (mode === "segmentation") {
        const result = await api.predictSegmentation(blob, modelId || null, tta);
        setSeg(result);
        setView("overlay");
      } else {
        const result = await api.predictClassification(blob, modelId || null);
        setClf(result);
        setView(result.gradcam_base64 ? "gradcam" : "original");
      }
      onPrediction();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed");
    } finally {
      setBusy(false);
    }
  };

  const sendFeedback = async (verdict: "correct" | "incorrect") => {
    const id = seg?.prediction_id ?? clf?.prediction_id;
    if (!id) return;
    try {
      await api.feedback(id, verdict);
      setFeedback(verdict);
      notify("Feedback recorded — it is now part of the monitoring ledger");
      onPrediction();
    } catch (err) {
      notify(err instanceof Error ? err.message : "Could not record feedback", "error");
    }
  };

  const views =
    mode === "segmentation"
      ? [
          { id: "original", label: "Original" },
          { id: "overlay", label: "Lesion overlay" },
          { id: "mask", label: "Hard mask" },
          { id: "uncertainty", label: "Uncertainty" },
        ]
      : [
          { id: "original", label: "Original" },
          { id: "gradcam", label: "Grad-CAM" },
        ];
  const activeView = views.some((item) => item.id === view) ? view : views[0].id;
  const result = seg ?? clf;
  const previewForResult = seg?.preview_base64 ?? clf?.preview_base64 ?? null;

  return (
    <section className="analyzer">
      <div className="column left">
        <div className="panel">
          <div className="panel-head">
            <div>
              <span className="eyebrow">Case intake</span>
              <h2>Diagnostic workstation</h2>
            </div>
            <ShieldIcon className="head-icon" />
          </div>

          <div className="segmented" role="tablist" aria-label="Task">
            <button
              type="button"
              role="tab"
              aria-selected={mode === "segmentation"}
              className={mode === "segmentation" ? "active" : ""}
              onClick={() => setMode("segmentation")}
            >
              <CrosshairIcon size={15} />
              Lesion segmentation
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === "classification"}
              className={mode === "classification" ? "active" : ""}
              onClick={() => setMode("classification")}
            >
              <BrainIcon size={15} />
              Chest X-ray triage
            </button>
          </div>

          <ImageDropzone
            previewUrl={previewUrl}
            filename={file?.name ?? demo?.name ?? null}
            disabled={busy}
            busy={busy}
            onFile={(picked) => {
              setFile(picked);
              setDemo(null);
              resetResults();
            }}
            onDemo={loadDemo}
            demoLoading={demoLoading}
          />

          <div className="settings">
            <label className="field">
              <span>Model</span>
              <select value={modelId} onChange={(event) => setModelId(event.target.value)} disabled={!taskModels.length}>
                <option value="">Auto · newest checkpoint</option>
                {taskModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.id} · {model.arch}
                  </option>
                ))}
              </select>
            </label>

            {mode === "segmentation" ? (
              <label className="toggle-row">
                <span>
                  Flip TTA
                  <small>Average prediction with horizontal flip + uncertainty map</small>
                </span>
                <button
                  type="button"
                  role="switch"
                  aria-checked={tta}
                  className={`switch ${tta ? "on" : ""}`}
                  onClick={() => setTta((value) => !value)}
                >
                  <span />
                </button>
              </label>
            ) : (
              <label className="toggle-row">
                <span>
                  Grad-CAM
                  <small>Saliency explanation is computed with every inference</small>
                </span>
                <span className="badge info">on</span>
              </label>
            )}
          </div>

          {!ready && (
            <div className="banner warn">
              <AlertIcon size={16} />
              <span>
                No {mode} checkpoint loaded. Train one first.{" "}
                <button type="button" className="link" onClick={() => onNavigate("experiments")}>
                  Open pipeline
                </button>
              </span>
            </div>
          )}

          <div className="actions">
            <button
              type="button"
              className="btn btn-primary"
              onClick={analyze}
              disabled={!previewUrl || busy || !ready}
            >
              {busy ? <span className="spinner" /> : <ScanIcon size={16} />}
              {busy ? "Analyzing…" : "Run inference"}
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => {
                setFile(null);
                setDemo(null);
                resetResults();
              }}
              disabled={busy || (!previewUrl && !result)}
            >
              <RefreshIcon size={15} />
              Reset
            </button>
          </div>

          {error && (
            <div className="banner error">
              <AlertIcon size={16} />
              <span>{error}</span>
            </div>
          )}

          <p className="fineprint">
            Images are processed in-memory. Outputs are assistive signals for research and must be reviewed by a
            clinician.
          </p>
        </div>
      </div>

      <div className="column right">
        <div className="panel viewer-panel">
          <div className="panel-head">
            <div>
              <span className="eyebrow">Inference viewer</span>
              <h2>{mode === "segmentation" ? "Lesion map" : "Saliency map"}</h2>
            </div>
            {result && (
              <span className="chip mono">
                {formatMs(result.latency_ms)} · {seg ? "U-Net + TTA" : clf?.model_id}
              </span>
            )}
          </div>

          <div className={`viewer-stage ${busy ? "scanning" : ""}`}>
            {previewForResult ? (
              <Viewer
                previewSrc={imgSrc(previewForResult)}
                maskSrc={seg ? imgSrc(seg.mask_base64) : null}
                overlaySrc={seg ? imgSrc(seg.overlay_base64) : null}
                uncertaintySrc={seg ? imgSrc(seg.uncertainty_base64) : null}
                gradcamSrc={clf?.gradcam_base64 ? imgSrc(clf.gradcam_base64) : null}
                view={activeView}
                opacity={opacity}
              />
            ) : previewUrl ? (
              <div className="viewer">
                <img src={previewUrl} alt="case preview" className="viewer-base" />
                {busy && <span className="scanline" />}
                <span className="corner tl" />
                <span className="corner tr" />
                <span className="corner bl" />
                <span className="corner br" />
              </div>
            ) : (
              <div className="viewer-placeholder">
                <CrosshairIcon size={38} />
                <p>No case loaded</p>
                <span>Attach a dermoscopy or chest X-ray image to begin.</span>
              </div>
            )}
          </div>

          <div className="viewer-toolbar">
            <div className="chips" role="group" aria-label="Layers">
              {views.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`chip-btn ${activeView === item.id ? "active" : ""}`}
                  onClick={() => setView(item.id)}
                  disabled={!result}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <label className="opacity">
              <span>Layer</span>
              <input
                type="range"
                min={10}
                max={100}
                value={Math.round(opacity * 100)}
                onChange={(event) => setOpacity(Number(event.target.value) / 100)}
                disabled={!result || activeView === "original"}
              />
              <span className="mono">{Math.round(opacity * 100)}%</span>
            </label>
          </div>

          {seg && (
            <>
              <div className="stat-grid">
                <StatCard
                  label="Lesion area"
                  value={formatPct(seg.lesion_fraction, 2)}
                  sub={`${Math.round(seg.lesion_fraction * seg.width * seg.height)} px`}
                  tone="accent"
                  icon={<CrosshairIcon size={16} />}
                />
                <StatCard label="Mean probability" value={formatPct(seg.mean_probability)} sub={`threshold ${seg.threshold.toFixed(2)}`} />
                <StatCard
                  label="Uncertainty"
                  value={seg.uncertainty_mean.toFixed(4)}
                  sub={seg.tta ? "flip-TTA disagreement" : "TTA disabled"}
                  tone={seg.uncertainty_mean > 0.08 ? "warn" : "ok"}
                />
                <StatCard label="Latency" value={formatMs(seg.latency_ms)} sub={`${seg.model_id}`} />
              </div>
              <FeedbackRow predictionId={seg.prediction_id} verdict={feedback} onFeedback={sendFeedback} />
            </>
          )}

          {clf && (
            <>
              <div className="clf-result">
                <div className="clf-headline">
                  <div>
                    <span className="eyebrow">Model verdict</span>
                    <h3>{clf.predicted_label.replace(/_/g, " ")}</h3>
                    <p className="muted">
                      {formatPct(clf.confidence)} confidence · margin {formatPct(clf.margin)} · {formatMs(clf.latency_ms)}
                    </p>
                  </div>
                  <svg viewBox="0 0 80 80" className="confidence-ring">
                    <circle cx="40" cy="40" r="32" className="ring-track" />
                    <circle
                      cx="40"
                      cy="40"
                      r="32"
                      className="ring-value"
                      strokeDasharray={`${clf.confidence * 201} 201`}
                    />
                    <text x="40" y="46" textAnchor="middle" className="ring-text">
                      {Math.round(clf.confidence * 100)}
                    </text>
                  </svg>
                </div>
                <div className="prob-bars">
                  {clf.labels.map((label, index) => (
                    <div className="prob-row" key={label}>
                      <span className="prob-label">{label.replace(/_/g, " ")}</span>
                      <div className="prob-track">
                        <div
                          className={`prob-fill ${index === clf.predicted_index ? "active" : ""}`}
                          style={{ width: `${Math.max(clf.probabilities[index] * 100, 1)}%` }}
                        />
                      </div>
                      <span className="prob-value mono">{formatPct(clf.probabilities[index])}</span>
                    </div>
                  ))}
                </div>
              </div>
              <FeedbackRow predictionId={clf.prediction_id} verdict={feedback} onFeedback={sendFeedback} />
            </>
          )}

          {result && (
            <div className="run-strip mono">
              <span>pred {result.prediction_id}</span>
              <span>run {result.model_version ? result.model_version.slice(0, 8) : "—"}</span>
              <span>{seg ? `tta ${String(seg.tta)}` : `classes ${clf?.labels.length}`}</span>
              <span>{seg ? seg.width : clf?.width}×{seg ? seg.height : clf?.height}</span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function FeedbackRow({
  predictionId,
  verdict,
  onFeedback,
}: {
  predictionId: string;
  verdict: "correct" | "incorrect" | null;
  onFeedback: (verdict: "correct" | "incorrect") => void;
}) {
  return (
    <div className="feedback-row">
      <span className="muted small">Clinician feedback for {predictionId}:</span>
      <div className="feedback-actions">
        <button
          type="button"
          className={`btn btn-ghost xs ${verdict === "correct" ? "selected-ok" : ""}`}
          onClick={() => onFeedback("correct")}
          disabled={verdict !== null}
        >
          <CheckIcon size={14} /> Agree
        </button>
        <button
          type="button"
          className={`btn btn-ghost xs ${verdict === "incorrect" ? "selected-alert" : ""}`}
          onClick={() => onFeedback("incorrect")}
          disabled={verdict !== null}
        >
          <XIcon size={14} /> Needs review
        </button>
        {verdict && <span className="badge ok">recorded</span>}
      </div>
    </div>
  );
}
