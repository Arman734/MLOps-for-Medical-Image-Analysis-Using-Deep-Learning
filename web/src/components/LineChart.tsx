import { useId } from "react";

export type Series = { label: string; color: string; points: { x: number; y: number }[] };

type Props = {
  series: Series[];
  height?: number;
  yFormat?: (value: number) => string;
};

export default function LineChart({ series, height = 220, yFormat = (v) => v.toFixed(2) }: Props) {
  const id = useId().replace(/:/g, "");
  const width = 640;
  const padding = { left: 46, right: 16, top: 16, bottom: 30 };
  const allX = series.flatMap((s) => s.points.map((p) => p.x));
  const allY = series.flatMap((s) => s.points.map((p) => p.y));
  if (!allX.length || !allY.length) return <div className="chart-empty">No metric history yet</div>;

  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const minY = Math.min(...allY);
  const maxY = Math.max(...allY);
  const padY = (maxY - minY) * 0.1 || 0.1;
  const loY = minY - padY;
  const hiY = maxY + padY;
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;

  const scaleX = (x: number) => padding.left + ((x - minX) / Math.max(maxX - minX, 1e-9)) * innerW;
  const scaleY = (y: number) => padding.top + innerH - ((y - loY) / (hiY - loY)) * innerH;

  const ticks = 4;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => loY + ((hiY - loY) * i) / ticks);
  const xTicks = Array.from({ length: 5 }, (_, i) => minX + ((maxX - minX) * i) / 4);

  return (
    <div className="line-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <defs>
          {series.map((s, i) => (
            <linearGradient key={i} id={`lc-${id}-${i}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={s.color} stopOpacity="0.24" />
              <stop offset="100%" stopColor={s.color} stopOpacity="0" />
            </linearGradient>
          ))}
        </defs>
        {yTicks.map((tick, i) => (
          <g key={i}>
            <line x1={padding.left} x2={width - padding.right} y1={scaleY(tick)} y2={scaleY(tick)} className="grid-line" />
            <text x={padding.left - 8} y={scaleY(tick) + 3} textAnchor="end" className="axis-text">
              {yFormat(tick)}
            </text>
          </g>
        ))}
        {xTicks.map((tick, i) => (
          <text key={i} x={scaleX(tick)} y={height - 8} textAnchor="middle" className="axis-text">
            {Math.round(tick)}
          </text>
        ))}
        {series.map((s, i) => {
          const path = s.points
            .map((p, j) => `${j === 0 ? "M" : "L"}${scaleX(p.x).toFixed(1)},${scaleY(p.y).toFixed(1)}`)
            .join(" ");
          const area = `${path} L${scaleX(s.points[s.points.length - 1].x).toFixed(1)},${padding.top + innerH} L${scaleX(s.points[0].x).toFixed(1)},${padding.top + innerH} Z`;
          return (
            <g key={s.label}>
              <path d={area} fill={`url(#lc-${id}-${i})`} />
              <path d={path} fill="none" stroke={s.color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              {s.points.map((p, j) => (
                <circle key={j} cx={scaleX(p.x)} cy={scaleY(p.y)} r="2.4" fill="#050b14" stroke={s.color} strokeWidth="1.6" />
              ))}
            </g>
          );
        })}
      </svg>
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.label} className="legend-item">
            <span className="legend-dot" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
