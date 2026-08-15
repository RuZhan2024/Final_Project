import type { PredictionRequest, SkeletonFrame } from "./api";

function frame(timestamp_ms: number, yOffset: number, height = 0.3): SkeletonFrame {
  return {
    timestamp_ms,
    keypoints: {
      head: { x: 0.5, y: yOffset, confidence: 0.95 },
      hip: { x: 0.5, y: yOffset + height / 2, confidence: 0.95 },
      ankle: { x: 0.5, y: yOffset + height, confidence: 0.95 },
    },
  };
}

export const sampleWindows = {
  stable: {
    source_id: "sample-stable",
    frames: [
      frame(0, 0.2),
      frame(500, 0.2),
      frame(1000, 0.2),
    ],
  },
  fall: {
    source_id: "sample-fall",
    frames: [
      frame(0, 0.1, 0.3),
      frame(500, 0.45, 0.2),
      frame(1000, 0.72, 0.1),
    ],
  },
} satisfies Record<string, PredictionRequest>;

export type SampleName = keyof typeof sampleWindows;
