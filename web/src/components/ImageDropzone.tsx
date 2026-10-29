import { useRef, useState } from "react";
import { ScanIcon, UploadIcon } from "./Icons";

type Props = {
  previewUrl: string | null;
  filename: string | null;
  disabled?: boolean;
  busy?: boolean;
  onFile: (file: File) => void;
  onDemo: () => void;
  demoLoading?: boolean;
};

export default function ImageDropzone({ previewUrl, filename, disabled, busy, onFile, onDemo, demoLoading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const pick = (files: FileList | null) => {
    if (!files || !files.length) return;
    const file = files[0];
    if (!file.type.startsWith("image/")) return;
    onFile(file);
  };

  return (
    <div className="dropzone-wrap">
      <div
        className={`dropzone ${dragging ? "dragging" : ""} ${previewUrl ? "has-image" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!disabled) pick(event.dataTransfer.files);
        }}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
        }}
        aria-label="Upload a medical image"
      >
        {previewUrl ? (
          <>
            <img src={previewUrl} alt={filename ?? "case preview"} className="dropzone-preview" />
            {busy && <span className="scanline" />}
            <div className="dropzone-overlay">
              <UploadIcon size={18} />
              <span>Replace image</span>
            </div>
          </>
        ) : (
          <div className="dropzone-empty">
            <div className="dropzone-icon">
              <ScanIcon size={26} />
            </div>
            <p className="dropzone-title">Drop a clinical image</p>
            <p className="dropzone-hint">Dermoscopy · chest X-ray · ultrasound — JPG or PNG up to 15 MB</p>
            <span className="btn btn-ghost sm">Browse files</span>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(event) => {
            pick(event.target.files);
            event.target.value = "";
          }}
        />
      </div>
      <div className="dropzone-actions">
        <span className="filename mono">{filename ?? "no case loaded"}</span>
        <button type="button" className="btn btn-ghost xs" onClick={onDemo} disabled={demoLoading || disabled}>
          {demoLoading ? "Loading…" : "Load demo case"}
        </button>
      </div>
    </div>
  );
}
