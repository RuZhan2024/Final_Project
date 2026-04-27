import type { MutableRefObject } from "react";

/**
 * Media-source helpers for monitor live camera and replay playback.
 *
 * These functions isolate browser media setup and teardown so hooks can treat
 * camera/replay preparation as one stable contract.
 */
export function syncReplayPlaybackRate(
  videoEl: HTMLVideoElement | null,
  latencyMs: number,
  inputSource: string
) {
  /** Adjust replay playback so UI video pace roughly follows backend throughput. */
  if (!videoEl || inputSource !== "video") return;

  let nextRate = 1.0;
  if (Number(latencyMs) >= 2200) nextRate = 0.9;

  if (Math.abs(Number(videoEl.playbackRate || 1) - nextRate) < 0.01) return;
  try {
    videoEl.playbackRate = nextRate;
  } catch {
    // ignore
  }
}

export function resetVideoSource({
  streamRef,
  videoRef,
  videoObjectUrlRef,
}: {
  streamRef: MutableRefObject<MediaStream | null>;
  videoRef: MutableRefObject<HTMLVideoElement | null>;
  videoObjectUrlRef: MutableRefObject<string | null>;
}) {
  /** Release any active camera stream or replay object URL before switching source. */
  if (streamRef.current) {
    streamRef.current.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  const videoEl = videoRef.current;
  if (videoEl) {
    try {
      videoEl.pause();
    } catch {
      // ignore
    }
    videoEl.srcObject = null;
    if (videoObjectUrlRef.current) {
      videoEl.removeAttribute("src");
      videoEl.load?.();
    }
  }

  if (videoObjectUrlRef.current) {
    try {
      URL.revokeObjectURL(videoObjectUrlRef.current);
    } catch {
      // ignore
    }
    videoObjectUrlRef.current = null;
  }
}

function getVideoLoadErrorMessage(videoEl: HTMLVideoElement | null, file: File | Blob | { type?: string } | null) {
  /** Turn browser media errors into a message that is useful for replay troubleshooting. */
  const code = Number(videoEl?.error?.code || 0);
  const byCode: Record<number, string> = {
    1: "Video loading aborted.",
    2: "Network/media fetch error while loading video.",
    3: "Video decode error (codec may be unsupported).",
    4: "Video format not supported by this browser.",
  };
  const base = byCode[code] || "Unknown video load error.";
  const typeHint = String(file?.type || "").trim();
  return `${base}${typeHint ? ` File type: ${typeHint}.` : ""} Try an H.264 MP4 (AAC) or WebM file.`;
}

export async function awaitVideoReady(videoEl: HTMLVideoElement, clipLabel: File | Blob | { type?: string } | null) {
  /**
   * Wait until the video element has enough metadata/data for pose processing.
   *
   * The monitor runtime requires readyState >= 2 before MediaPipe can safely
   * consume frames from the element.
   */
  await new Promise<void>((resolve, reject) => {
    if (videoEl.readyState >= 2) {
      resolve();
      return;
    }

    let done = false;
    const cleanup = () => {
      videoEl.removeEventListener("loadedmetadata", onReady);
      videoEl.removeEventListener("loadeddata", onReady);
      videoEl.removeEventListener("canplay", onReady);
      videoEl.removeEventListener("error", onError);
      window.clearTimeout(timerId);
    };

    const onReady = () => {
      if (done || videoEl.readyState < 2) return;
      done = true;
      cleanup();
      resolve();
    };

    const onError = () => {
      if (done) return;
      done = true;
      cleanup();
      reject(new Error(getVideoLoadErrorMessage(videoEl, clipLabel)));
    };

    const timerId = window.setTimeout(() => {
      if (done) return;
      done = true;
      cleanup();
      reject(new Error("Video metadata load timeout"));
    }, 10000);

    videoEl.addEventListener("loadedmetadata", onReady, { once: true });
    videoEl.addEventListener("loadeddata", onReady, { once: true });
    videoEl.addEventListener("canplay", onReady, { once: true });
    videoEl.addEventListener("error", onError, { once: true });

    try {
      videoEl.load?.();
    } catch {
      // ignore
    }
  });
}

export async function prepareReplayVideo({
  videoEl,
  clip,
  clipUrl,
  videoObjectUrlRef,
  setStartInfo,
  fetchReplayClipBlob,
  resetVideoSource,
}: {
  videoEl: HTMLVideoElement;
  clip: any;
  clipUrl: string;
  videoObjectUrlRef: MutableRefObject<string | null>;
  setStartInfo: (value: string) => void;
  fetchReplayClipBlob: (clipUrl: string) => Promise<Blob>;
  resetVideoSource: () => void;
}) {
  /** Load a replay file/URL into the shared video element and reset playback state. */
  const clipFile = clip?.file instanceof File ? clip.file : null;

  resetVideoSource();
  if (clipFile) {
    // Local files bypass the backend entirely, but still use an object URL so
    // the rest of the replay pipeline can treat file and fetched clips the same.
    const objectUrl = URL.createObjectURL(clipFile);
    videoObjectUrlRef.current = objectUrl;
    videoEl.src = objectUrl;
  } else {
    setStartInfo("Loading replay clip...");
    const blob = await fetchReplayClipBlob(clipUrl);
    const objectUrl = URL.createObjectURL(blob);
    videoObjectUrlRef.current = objectUrl;
    videoEl.src = objectUrl;
  }

  videoEl.currentTime = 0;
  videoEl.muted = true;
  videoEl.playsInline = true;
  videoEl.preload = "auto";

  await awaitVideoReady(videoEl, clipFile || clip);
}

export async function prepareCameraStream({
  videoEl,
  captureResolution,
  targetFps,
  streamRef,
}: {
  videoEl: HTMLVideoElement;
  captureResolution: { w: number; h: number };
  targetFps: number;
  streamRef: MutableRefObject<MediaStream | null>;
}) {
  /** Start the camera stream with the requested capture resolution and FPS target. */
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      // These are ideals/maxima rather than hard constraints so browsers can
      // still negotiate a workable camera mode on limited devices.
      width: { ideal: captureResolution.w, max: captureResolution.w },
      height: { ideal: captureResolution.h, max: captureResolution.h },
      frameRate: { ideal: targetFps, max: targetFps },
    },
    audio: false,
  });
  streamRef.current = stream;
  videoEl.srcObject = stream;
  await awaitVideoReady(videoEl, null);
}
