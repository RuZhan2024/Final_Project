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
  if (!rawFrames || rawFrames.length < 2) return false;

  const dtMs = 1000 / Math.max(1, Number(targetFps) || 30);
  const needMs = (deployW - 1) * dtMs;
  const strideMs = Math.max(dtMs, Number(deployS) * dtMs);
  const endTs = rawFrames[rawFrames.length - 1].t;
  const firstEligibleEndTs = rawFrames[0].t + needMs;
  if (!Number.isFinite(firstEligibleEndTs) || endTs < firstEligibleEndTs) return false;

  if (nextWindowEndRef.current == null || nextWindowEndRef.current < firstEligibleEndTs) {
    nextWindowEndRef.current = firstEligibleEndTs;
  }

  let queuedAny = false;
  while (Number(nextWindowEndRef.current) <= endTs) {
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
  if (!rawFrames || rawFrames.length < 2) return null;

  const dtMs = 1000 / Math.max(1, Number(targetFps) || 30);
  const needMs = (deployW - 1) * dtMs;
  const startWindowTs = windowEndTs - needMs;
  if (rawFrames[0].t > startWindowTs) return null;

  let startIndex = 0;
  while (startIndex < rawFrames.length && rawFrames[startIndex].t < startWindowTs) startIndex += 1;
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
  if (!rawFrames || rawFrames.length < 2) return null;

  const dtMs = 1000 / Math.max(1, Number(targetFps) || 30);
  const needMs = (deployW - 1) * dtMs;
  const endTs = rawFrames[rawFrames.length - 1].t;
  const startNeed = endTs - needMs;
  if (rawFrames[0].t > startNeed) return null;

  let startIndex = 0;
  while (startIndex < rawFrames.length && rawFrames[startIndex].t < startNeed) startIndex += 1;
  return {
    endTs,
    frames: rawFrames.slice(Math.max(0, startIndex - 1)),
  };
}
