import { MONITOR_PAYLOAD_Q_SCALE, NUM_JOINTS } from "../../pages/monitor/constants";
import { clamp01 } from "../../pages/monitor/utils";

export const TRIAGE_CONFIRMATION_COUNTS = {
  fall: 2,
  safe: 2,
  uncertain: 3,
};

export const FALL_HISTORY_DEDUP_MS_DEFAULT = 30_000;

export function buildQuantizedWindow(slice) {
  const nSlice = Array.isArray(slice) ? slice.length : 0;
  const tArr = new Array(nSlice);
  const xyQ = new Array(nSlice * NUM_JOINTS * 2);
  const confQ = new Array(nSlice * NUM_JOINTS);

  let q = 0;
  let c = 0;
  for (let i = 0; i < nSlice; i += 1) {
    const frame = slice[i] || {};
    tArr[i] = frame.t;
    const pts = frame.xy || [];
    const conf = frame.conf || [];
    for (let j = 0; j < NUM_JOINTS; j += 1) {
      const point = pts[j] || [0, 0];
      const visibility = conf[j] == null ? 1 : conf[j];
      xyQ[q++] = Math.round((Number.isFinite(Number(point[0])) ? Number(point[0]) : 0) * MONITOR_PAYLOAD_Q_SCALE);
      xyQ[q++] = Math.round((Number.isFinite(Number(point[1])) ? Number(point[1]) : 0) * MONITOR_PAYLOAD_Q_SCALE);
      confQ[c++] = Math.round(clamp01(Number(visibility) || 0) * MONITOR_PAYLOAD_Q_SCALE);
    }
  }

  return { nSlice, tArr, xyQ, confQ };
}

export function buildPredictPayload({
  slice,
  sessionId,
  windowSeq,
  inputSource,
  location,
  mode,
  datasetCode,
  opCode,
  chosen,
  targetFps,
  deployW,
  streamFps,
  mcEnabled,
  mcCfg,
  persist,
  endTs,
}) {
  const { nSlice, tArr, xyQ, confQ } = buildQuantizedWindow(slice);
  return {
    window_seq: windowSeq,
    session_id: sessionId,
    input_source: inputSource,
    resident_id: 1,
    location,
    mode,
    dataset_code: datasetCode,
    op_code: opCode,
    model_tcn: mode !== "gcn" ? chosen.tcn : null,
    model_gcn: mode !== "tcn" ? chosen.gcn : null,
    model_id: mode === "tcn" ? chosen.tcn : mode === "gcn" ? chosen.gcn : null,
    fps: targetFps,
    target_fps: targetFps,
    target_T: deployW,
    capture_fps: streamFps || null,
    timestamp_ms: Date.now(),
    use_mc: Boolean(mcEnabled),
    mc_M: mcCfg?.M,
    persist: Boolean(persist),
    raw_t_ms: tArr,
    raw_shape: [nSlice, NUM_JOINTS],
    raw_xy_q: xyQ,
    raw_conf_q: confQ,
    window_end_t_ms: endTs,
  };
}

export function resolveStableTriage(previousStable, triageCandidate, safeAlert) {
  const next = {
    fall: Number(previousStable?.fall || 0),
    uncertain: Number(previousStable?.uncertain || 0),
    safe: Number(previousStable?.safe || 0),
    last: String(previousStable?.last || "not_fall"),
  };

  if (triageCandidate === "fall" && safeAlert !== false) {
    next.fall += 1;
    next.uncertain = 0;
    next.safe = 0;
  } else if (triageCandidate === "uncertain") {
    next.uncertain += 1;
    next.fall = 0;
    next.safe = 0;
  } else {
    next.safe += 1;
    next.fall = 0;
    next.uncertain = 0;
  }

  let triageState = next.last || "not_fall";
  if (next.fall >= TRIAGE_CONFIRMATION_COUNTS.fall) triageState = "fall";
  else if (next.safe >= TRIAGE_CONFIRMATION_COUNTS.safe) triageState = "not_fall";
  else if (next.uncertain >= TRIAGE_CONFIRMATION_COUNTS.uncertain && next.fall === 0) triageState = "uncertain";

  next.last = triageState;
  return { stable: next, triageState };
}

export function extractPredictionState({ data, mode, previousStable, settingsPayload }) {
  const safeObj = data?.policy_alerts?.safe;
  const recallObj = data?.policy_alerts?.recall;
  const triRaw = String(data?.triage_state || data?.triageState || "not_fall").toLowerCase();
  const safeStateRaw = String(data?.safe_state || safeObj?.state || "").toLowerCase();
  const safeAlert =
    typeof data?.safe_alert === "boolean"
      ? data.safe_alert
      : typeof safeObj?.alert === "boolean"
        ? safeObj.alert
        : null;
  const recallAlert =
    typeof data?.recall_alert === "boolean"
      ? data.recall_alert
      : typeof recallObj?.alert === "boolean"
        ? recallObj.alert
        : null;
  const safeState = (data?.safe_state || safeObj?.state || null) ?? null;
  const recallState = (data?.recall_state || recallObj?.state || null) ?? null;

  let triageCandidate = safeStateRaw || triRaw || "not_fall";
  if (triageCandidate !== "fall" && triageCandidate !== "uncertain") triageCandidate = "not_fall";
  const { stable, triageState } = resolveStableTriage(previousStable, triageCandidate, safeAlert);

  const modelOutput = mode === "hybrid" ? data?.models?.tcn : data?.models?.[mode];
  const pFall = modelOutput
    ? modelOutput?.triage?.ps != null
      ? Number(modelOutput.triage.ps)
      : modelOutput?.p_alert_in != null
        ? Number(modelOutput.p_alert_in)
        : modelOutput?.mu != null
          ? Number(modelOutput.mu)
          : modelOutput?.p_det != null
            ? Number(modelOutput.p_det)
            : null
    : null;
  const sigma = modelOutput?.sigma != null ? Number(modelOutput.sigma) : null;

  let markerKind = "safe";
  if (triageState === "fall") markerKind = "fall";
  else if (triageState === "uncertain") markerKind = "uncertain";

  const dedupMs = Math.round(
    Math.max(
      1000,
      1000 * Number(settingsPayload?.system?.alert_cooldown_sec || FALL_HISTORY_DEDUP_MS_DEFAULT / 1000)
    )
  );

  return {
    safeAlert,
    recallAlert,
    safeState,
    recallState,
    stable,
    triageState,
    pFall,
    sigma,
    markerKind,
    dedupMs,
    eventId: data?.event_id ?? null,
  };
}

export function buildPendingClip({
  eventId,
  endTs,
  activeDatasetCode,
  mode,
  opCode,
  settingsPayload,
  mcEnabled,
  mcCfg,
  clipPreS,
  clipPostS,
}) {
  return {
    eventId,
    triggerEndTs: endTs,
    deadlineTs: endTs + clipPostS * 1000,
    preMs: clipPreS * 1000,
    postMs: clipPostS * 1000,
    ctx: {
      dataset_code: activeDatasetCode,
      mode,
      op_code: opCode || settingsPayload?.system?.active_op_code || "OP-1",
      use_mc: Boolean(mcEnabled),
      mc_M: mcCfg?.M,
    },
  };
}
