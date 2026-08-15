export type Keypoint = {
  x: number;
  y: number;
  confidence: number;
};

export type SkeletonFrame = {
  timestamp_ms: number;
  keypoints: Record<string, Keypoint>;
};

export type PredictionRequest = {
  source_id?: string;
  frames: SkeletonFrame[];
};

export type PredictionResponse = {
  label: "fall" | "non_fall";
  fall_probability: number;
  confidence: number;
  model_name: string;
  reasons: string[];
};

export async function requestPrediction(
  apiBaseUrl: string,
  payload: PredictionRequest,
): Promise<PredictionResponse> {
  const response = await fetch(`${apiBaseUrl}/api/v1/predictions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Prediction request failed with ${response.status}`);
  }

  return response.json() as Promise<PredictionResponse>;
}
