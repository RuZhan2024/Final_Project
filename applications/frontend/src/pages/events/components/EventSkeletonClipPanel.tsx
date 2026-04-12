import React, { useEffect, useMemo, useRef, useState } from "react";

import { fetchEventSkeletonClip } from "../../../features/events/api";
import type { EventRecord, EventSkeletonClipResponse } from "../../../features/events/types";
import { parseDateSafe } from "../../../lib/dates";

import styles from "../../Events.module.css";

const SKELETON_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31],
  [24, 26], [26, 28], [28, 30], [30, 32],
];

function hasStoredSkeletonClip(event: EventRecord | null): boolean {
  return Boolean(event?.meta?.skeleton_clip?.path);
}

interface EventSkeletonClipPanelProps {
  apiBase: string;
  event: EventRecord | null;
  residentId?: number;
}

export function EventSkeletonClipPanel({
  apiBase,
  event,
  residentId = 1,
}: EventSkeletonClipPanelProps) {
  const [clipLoading, setClipLoading] = useState(false);
  const [clipError, setClipError] = useState("");
  const [clipData, setClipData] = useState<EventSkeletonClipResponse | null>(null);
  const [clipFrameIdx, setClipFrameIdx] = useState(0);
  const [clipPlaying, setClipPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(0.75);
  const clipCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let active = true;
    const ac = new AbortController();

    async function loadClip() {
      if (!event || !hasStoredSkeletonClip(event)) return;
      try {
        setClipLoading(true);
        setClipError("");
        const data = await fetchEventSkeletonClip(apiBase, event.id, {
          residentId,
          signal: ac.signal,
        });
        if (!active) return;
        setClipData(data || null);
        setClipFrameIdx(0);
      } catch (e: unknown) {
        if (!active || (e as Error)?.name === "AbortError") return;
        setClipError(String((e as Error)?.message || e));
      } finally {
        if (active) setClipLoading(false);
      }
    }

    setClipLoading(false);
    setClipError("");
    setClipData(null);
    setClipFrameIdx(0);
    setClipPlaying(false);
    void loadClip();

    return () => {
      active = false;
      ac.abort();
    };
  }, [apiBase, event, residentId]);

  useEffect(() => {
    if (!clipPlaying || !clipData?.t_ms?.length) return undefined;
    const tMs = Array.isArray(clipData.t_ms) ? clipData.t_ms : [];
    if (tMs.length < 2) return undefined;

    const currentIdx = Math.max(0, Math.min(clipFrameIdx, tMs.length - 1));
    if (currentIdx >= tMs.length - 1) {
      setClipPlaying(false);
      return undefined;
    }

    const currentT = Number(tMs[currentIdx]) || 0;
    const nextT = Number(tMs[currentIdx + 1]) || currentT + 42;
    const speed = Number(playbackRate) > 0 ? Number(playbackRate) : 0.75;
    const delayMs = Math.max(16, Math.min(200, (nextT - currentT) / speed));
    const timer = window.setTimeout(() => {
      setClipFrameIdx((prev) => {
        if (prev >= tMs.length - 1) {
          setClipPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, delayMs);
    return () => window.clearTimeout(timer);
  }, [clipPlaying, clipData, clipFrameIdx, playbackRate]);

  useEffect(() => {
    const canvas = clipCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#F9FAFB";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const xyFrames = Array.isArray(clipData?.xy) ? clipData.xy : [];
    const confFrames = Array.isArray(clipData?.conf) ? clipData.conf : [];
    if (!xyFrames.length) {
      ctx.fillStyle = "#6B7280";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No skeleton clip available", canvas.width / 2, canvas.height / 2);
      return;
    }

    const frame = xyFrames[Math.max(0, Math.min(clipFrameIdx, xyFrames.length - 1))] || [];
    const conf = confFrames[Math.max(0, Math.min(clipFrameIdx, confFrames.length - 1))] || [];
    const points: Array<{ idx: number; x: number; y: number }> = [];
    for (let i = 0; i < frame.length; i += 1) {
      const pair = frame[i];
      const c = Number(conf[i] ?? 1);
      if (!Array.isArray(pair) || pair.length < 2 || c < 0.05) continue;
      const x = Number(pair[0]);
      const y = Number(pair[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      points.push({ idx: i, x, y });
    }

    if (!points.length) {
      ctx.fillStyle = "#6B7280";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Stored clip has no visible joints", canvas.width / 2, canvas.height / 2);
      return;
    }

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    points.forEach((p) => {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    });

    const useNormalizedCoords =
      minX >= -0.05 && maxX <= 1.05 && minY >= -0.05 && maxY <= 1.05;

    const byIdx = new Map<number, { x: number; y: number }>();
    points.forEach((p) => {
      if (useNormalizedCoords) {
        const pad = 18;
        const drawW = canvas.width - pad * 2;
        const drawH = canvas.height - pad * 2;
        byIdx.set(p.idx, {
          x: pad + p.x * drawW,
          y: pad + p.y * drawH,
        });
        return;
      }

      const spanX = Math.max(1e-3, maxX - minX);
      const spanY = Math.max(1e-3, maxY - minY);
      const scale = Math.min((canvas.width - 48) / spanX, (canvas.height - 48) / spanY);
      const offsetX = (canvas.width - spanX * scale) / 2;
      const offsetY = (canvas.height - spanY * scale) / 2;
      byIdx.set(p.idx, {
        x: offsetX + (p.x - minX) * scale,
        y: offsetY + (p.y - minY) * scale,
      });
    });

    ctx.strokeStyle = "#2563EB";
    ctx.lineWidth = 2;
    SKELETON_EDGES.forEach(([a, b]) => {
      const pa = byIdx.get(a);
      const pb = byIdx.get(b);
      if (!pa || !pb) return;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    });

    ctx.fillStyle = "#111827";
    byIdx.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [clipData, clipFrameIdx]);

  const clipFrameCount = Array.isArray(clipData?.t_ms) ? clipData.t_ms.length : 0;
  const clipSavedAt = useMemo(() => {
    return clipData?.clip?.saved_at ? parseDateSafe(clipData.clip.saved_at) : null;
  }, [clipData]);
  const clipSourceMode = String(clipData?.clip?.mode || "—");
  const clipSourceOpCode = String(clipData?.clip?.op_code || "—");

  return (
    <div className={styles.clipPanel}>
      <div className={styles.clipPanelHeader}>
        <div>
          <h4 className={styles.clipTitle}>Stored Skeleton Replay</h4>
          <p className={styles.clipMeta}>
            {hasStoredSkeletonClip(event)
              ? "Event-linked skeleton clip captured around the detected incident."
              : "No stored skeleton clip is attached to this event."}
          </p>
        </div>
        {clipData?.clip ? (
          <div className={styles.clipStats}>
            <span>{clipData.clip.n_frames || clipFrameCount} frames</span>
            <span>{clipData.clip.anonymized ? "Anonymized" : "Raw skeleton"}</span>
          </div>
        ) : null}
      </div>

      <div className={styles.clipCanvasWrap}>
        <canvas ref={clipCanvasRef} width={420} height={260} className={styles.clipCanvas} />
      </div>

      {clipLoading ? <div className={styles.clipMessage}>Loading stored skeleton clip…</div> : null}
      {!clipLoading && clipError ? <div className={styles.clipError}>Clip error: {clipError}</div> : null}
      {!clipLoading && !clipError && clipData?.clip ? (
        <>
          <div className={styles.clipControls}>
            <button className={styles.secondaryBtn} onClick={() => setClipPlaying((v) => !v)}>
              {clipPlaying ? "Pause" : "Play"}
            </button>
            <button
              className={styles.secondaryBtn}
              onClick={() => {
                setClipPlaying(false);
                setClipFrameIdx(0);
              }}
            >
              Restart
            </button>
            <label className={styles.clipSpeedLabel}>
              Speed
              <select
                className={styles.clipSpeedSelect}
                value={String(playbackRate)}
                onChange={(e) => setPlaybackRate(Number(e.target.value) || 0.75)}
              >
                <option value="0.5">0.5x</option>
                <option value="0.75">0.75x</option>
                <option value="1">1.0x</option>
              </select>
            </label>
            <span className={styles.clipFrameLabel}>
              Frame {Math.min(clipFrameIdx + 1, clipFrameCount || 0)} / {clipFrameCount || 0}
            </span>
          </div>
          <input
            className={styles.clipSlider}
            type="range"
            min={0}
            max={Math.max(0, clipFrameCount - 1)}
            value={Math.min(clipFrameIdx, Math.max(0, clipFrameCount - 1))}
            onChange={(e) => {
              setClipPlaying(false);
              setClipFrameIdx(Number(e.target.value) || 0);
            }}
          />
          <div className={styles.clipMetaRow}>
            <span>{`Source: ${clipSourceMode} / ${clipSourceOpCode}`}</span>
            <span>{clipSavedAt ? clipSavedAt.toLocaleString() : "Saved clip"}</span>
          </div>
        </>
      ) : null}
    </div>
  );
}
