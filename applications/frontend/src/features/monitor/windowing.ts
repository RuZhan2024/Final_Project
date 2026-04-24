/**
 * Time-window helpers shared by live and replay monitor prediction.
 *
 * These helpers convert raw timestamped pose frames into the overlapping window
 * boundaries expected by the backend monitor pipeline.
 */
export function queueReplayWindowEnds({
  rawFrames,
  targetFps,
  deployW,
  deployS,
  nextWindowEndRef,
  queueRef,
}: {
  rawFrames: Array<{ t: number }>;
  targetFps: number;
  deployW: number;
  deployS: number;
  nextWindowEndRef: { current: number | null };
  queueRef: { current: number[] };
}) {
  /** Queue every replay window end time that is now fully covered by raw frames. */
  if (!rawFrames || rawFrames.length < 2) return false;

  const dtMs = 1000 / Math.max(1, Number(targetFps) || 30);
  const needMs = (deployW - 1) * dtMs;
  const strideMs = Math.max(dtMs, Number(deployS) * dtMs);
  const endTs = rawFrames[rawFrames.length - 1].t;
  const firstEligibleEndTs = rawFrames[0].t + needMs;
  if (!Number.isFinite(firstEligibleEndTs) || endTs < firstEligibleEndTs) return false;

  if (nextWindowEndRef.current == null || nextWindowEndRef.current < firstEligibleEndTs) {
    // The first queued replay window must be fully covered; starting earlier
    // would create a short window the backend interprets as missing context.
    nextWindowEndRef.current = firstEligibleEndTs;
  }

  let queuedAny = false;
  while (Number(nextWindowEndRef.current) <= endTs) {
    // Queue by end timestamp instead of frame index so replay slicing remains
    // stable even when capture timestamps are slightly irregular.
    queueRef.current.push(Number(nextWindowEndRef.current));
    nextWindowEndRef.current += strideMs;
    queuedAny = true;
  }
  return queuedAny;
}

export function sliceReplayWindowFrames<T extends { t: number }>({
  rawFrames,
  targetFps,
  deployW,
  windowEndTs,
}: {
  rawFrames: T[];
  targetFps: number;
  deployW: number;
  windowEndTs: number;
}) {
  /** Slice the raw frame buffer around one queued replay window end timestamp. */
  if (!rawFrames || rawFrames.length < 2) return null;

  const dtMs = 1000 / Math.max(1, Number(targetFps) || 30);
  const needMs = (deployW - 1) * dtMs;
  const startWindowTs = windowEndTs - needMs;
  // Reject incomplete windows here so replay and live modes share the same
  // "fully covered window only" contract before the backend sees any payload.
  if (rawFrames[0].t > startWindowTs) return null;

  let startIndex = 0;
  while (startIndex < rawFrames.length && rawFrames[startIndex].t < startWindowTs) startIndex += 1;
  // Include one frame before the nominal boundary so backend resampling still
  // has interpolation context when the exact window start is between frames.
  const boundedStart = Math.max(0, startIndex - 1);

  let endIndex = rawFrames.length;
  while (endIndex > boundedStart + 1 && rawFrames[endIndex - 1].t > windowEndTs) endIndex -= 1;
  return rawFrames.slice(boundedStart, endIndex);
}

export function sliceLiveWindowFrames<T extends { t: number }>({
  rawFrames,
  targetFps,
  deployW,
}: {
  rawFrames: T[];
  targetFps: number;
  deployW: number;
}) {
  /** Slice the latest live window ending at the newest buffered raw frame. */
  if (!rawFrames || rawFrames.length < 2) return null;

  const dtMs = 1000 / Math.max(1, Number(targetFps) || 30);
  const needMs = (deployW - 1) * dtMs;
  const endTs = rawFrames[rawFrames.length - 1].t;
  const startNeed = endTs - needMs;
  // Live mode also waits for full coverage; otherwise the backend would infer
  // on partially warmed-up windows and oscillate at session start.
  if (rawFrames[0].t > startNeed) return null;

  let startIndex = 0;
  while (startIndex < rawFrames.length && rawFrames[startIndex].t < startNeed) startIndex += 1;
  return {
    endTs,
    // Live mode keeps one earlier frame for the same interpolation reason used
    // by replay slicing; both paths should feed equivalent backend resampling.
    frames: rawFrames.slice(Math.max(0, startIndex - 1)),
  };
}
