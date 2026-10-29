import { useEffect, useRef } from "react";

type MaskLayerProps = { maskSrc: string; opacity: number; color?: [number, number, number] };

function MaskLayer({ maskSrc, opacity, color = [34, 211, 238] }: MaskLayerProps) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let cancelled = false;
    const image = new Image();
    image.onload = () => {
      if (cancelled) return;
      const canvas = ref.current;
      if (!canvas) return;
      canvas.width = image.width;
      canvas.height = image.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(image, 0, 0);
      const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < data.data.length; i += 4) {
        const value = data.data[i];
        data.data[i] = color[0];
        data.data[i + 1] = color[1];
        data.data[i + 2] = color[2];
        data.data[i + 3] = value > 127 ? 255 : 0;
      }
      ctx.putImageData(data, 0, 0);
    };
    image.src = maskSrc;
    return () => {
      cancelled = true;
    };
  }, [maskSrc, color]);

  return <canvas ref={ref} className="viewer-layer" style={{ opacity }} aria-hidden />;
}

type Props = {
  previewSrc: string;
  maskSrc?: string | null;
  overlaySrc?: string | null;
  uncertaintySrc?: string | null;
  gradcamSrc?: string | null;
  view: string;
  opacity: number;
};

export default function Viewer({ previewSrc, maskSrc, overlaySrc, uncertaintySrc, gradcamSrc, view, opacity }: Props) {
  return (
    <div className="viewer">
      <img src={previewSrc} alt="case" className="viewer-base" draggable={false} />
      {view === "mask" && maskSrc && <MaskLayer maskSrc={maskSrc} opacity={opacity} />}
      {view === "overlay" && overlaySrc && <img src={overlaySrc} alt="model overlay" className="viewer-layer" style={{ opacity }} draggable={false} />}
      {view === "uncertainty" && uncertaintySrc && (
        <img src={uncertaintySrc} alt="predictive uncertainty" className="viewer-layer" style={{ opacity }} draggable={false} />
      )}
      {view === "gradcam" && gradcamSrc && <img src={gradcamSrc} alt="grad-cam saliency" className="viewer-layer" style={{ opacity }} draggable={false} />}
      <span className="corner tl" />
      <span className="corner tr" />
      <span className="corner bl" />
      <span className="corner br" />
    </div>
  );
}
