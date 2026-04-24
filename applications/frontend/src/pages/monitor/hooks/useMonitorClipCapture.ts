import { useCallback, useRef } from "react";

import { uploadSkeletonClip } from "../../../features/monitor/api";
import { buildPendingClip } from "../../../features/monitor/prediction";
import type { PendingClip, UseMonitorClipCaptureOptions } from "../types";

function getClipFlags(settingsPayload: UseMonitorClipCaptureOptions["settingsPayload"]) {
  // Clip storage is opt-in; anonymization stays on unless settings disable it explicitly.
  const sys = settingsPayload?.system || {};
  return {
    storeEventClips: Boolean(sys?.store_event_clips),
    anonymize: sys?.anonymize_skeleton_data !== false,
  };
}

/**
 * Manages delayed skeleton-clip upload for persisted fall events.
 *
 * The backend creates the event id first; this hook waits until enough post-roll
 * frames exist, then uploads a skeleton-only clip tied to that event.
 */
export function useMonitorClipCapture({
  apiBase,
  settingsPayload,
  rawFramesRef,
  activeDatasetCode,
  mode,
  opCode,
  mcEnabled,
  mcCfg,
}: UseMonitorClipCaptureOptions) {
  const pendingClipRef = useRef<PendingClip | null>(null);
  const pendingClipTimerRef = useRef<number | null>(null);
  const uploadedClipIdsRef = useRef<Set<string>>(new Set());
  const clipFlags = getClipFlags(settingsPayload);

  const clearPendingClipTimer = useCallback(() => {
    /** Cancel any scheduled forced finalization of the pending clip upload. */
    const timerId = pendingClipTimerRef.current;
    if (timerId != null) {
      window.clearTimeout(timerId);
      pendingClipTimerRef.current = null;
    }
  }, []);

  const resetClipCaptureState = useCallback(() => {
    /** Drop pending/uploaded clip tracking for a fresh monitor session. */
    pendingClipRef.current = null;
    clearPendingClipTimer();
    uploadedClipIdsRef.current = new Set();
  }, [clearPendingClipTimer]);

  const maybeFinalizeClipUpload = useCallback(
    async ({ force = false }: { force?: boolean } = {}) => {
      /**
       * Upload the pending skeleton clip once enough post-event context exists.
       *
       * Normal completion waits until the pending deadline. Forced completion is
       * used when the timer fires or the session is ending and we want the best
       * available clip rather than losing the event attachment entirely.
       */
      const { storeEventClips } = getClipFlags(settingsPayload);
      if (!storeEventClips) {
        pendingClipRef.current = null;
        clearPendingClipTimer();
        return;
      }

      const pending = pendingClipRef.current;
      if (!pending) return;

      const raw = rawFramesRef.current;
      if (!raw || raw.length < 2) return;

      const lastT = raw[raw.length - 1]?.t;
      if (typeof lastT !== "number") return;
      if (!force && lastT < pending.deadlineTs) return;

      const sent = uploadedClipIdsRef.current;
      if (sent.has(String(pending.eventId))) {
        pendingClipRef.current = null;
        clearPendingClipTimer();
        return;
      }

      const startTs = pending.triggerEndTs - pending.preMs;
      const endTs = force ? Math.max(pending.triggerEndTs, lastT) : pending.triggerEndTs + pending.postMs;
      // Slice by timestamps instead of array position so live and replay buffers
      // can trim older frames without breaking clip alignment.
      const frames = raw.filter((fr) => fr && typeof fr.t === "number" && fr.t >= startTs && fr.t <= endTs);
      if (!frames || frames.length < 2) {
        pendingClipRef.current = null;
        clearPendingClipTimer();
        return;
      }

      try {
        const nFrames = frames.length;
        const tMs = new Array<number>(nFrames);
        const xy = new Array<number[][]>(nFrames);
        const conf = new Array<number[]>(nFrames);
        for (let i = 0; i < nFrames; i += 1) {
          const fr = frames[i];
          tMs[i] = fr.t;
          xy[i] = fr.xy;
          conf[i] = fr.conf;
        }

        const clipPayload = {
          // Resident settings remain single-resident today, so clip upload uses the same default id.
          resident_id: 1,
          dataset_code: pending.ctx?.dataset_code,
          mode: pending.ctx?.mode,
          op_code: pending.ctx?.op_code,
          use_mc: pending.ctx?.use_mc,
          mc_M: pending.ctx?.mc_M,
          pre_s: pending.preMs / 1000,
          post_s: pending.postMs / 1000,
          t_ms: tMs,
          xy,
          conf,
        };

        const data = await uploadSkeletonClip(apiBase, pending.eventId, clipPayload);
        if ((data as { ok?: boolean })?.ok) {
          sent.add(String(pending.eventId));
        }
      } catch (err) {
        // Clip upload failure should not break the live monitor loop.
        console.error("Failed to upload skeleton clip", err);
      } finally {
        pendingClipRef.current = null;
        clearPendingClipTimer();
      }
    },
    [apiBase, clearPendingClipTimer, rawFramesRef, settingsPayload]
  );

  const schedulePendingClipFinalize = useCallback(
    (pending: PendingClip | null) => {
      /** Schedule a forced upload shortly after the requested post-roll window. */
      clearPendingClipTimer();
      if (!pending) return;
      const delayMs = Math.max(500, Math.round(Number(pending.postMs || 0) + 250));
      pendingClipTimerRef.current = window.setTimeout(() => {
        void maybeFinalizeClipUpload({ force: true });
      }, delayMs);
    },
    [clearPendingClipTimer, maybeFinalizeClipUpload]
  );

  const queueClipForEvent = useCallback(
    ({
      eventId,
      endTs,
      clipPreS,
      clipPostS,
    }: {
      eventId: number | string;
      endTs: number;
      clipPreS: number;
      clipPostS: number;
    }) => {
      /** Create/replace the pending clip plan for a newly observed event id. */
      const nextPending = buildPendingClip({
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
      });
      pendingClipRef.current = nextPending;
      schedulePendingClipFinalize(nextPending);
    },
    [activeDatasetCode, mcCfg, mcEnabled, mode, opCode, schedulePendingClipFinalize, settingsPayload]
  );

  return {
    clipFlags,
    pendingClipRef,
    uploadedClipIdsRef,
    clearPendingClipTimer,
    resetClipCaptureState,
    maybeFinalizeClipUpload,
    schedulePendingClipFinalize,
    queueClipForEvent,
  };
}
