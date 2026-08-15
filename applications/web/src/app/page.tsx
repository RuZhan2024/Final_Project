"use client";

import { Activity, CheckCircle2, ShieldAlert, Siren } from "lucide-react";
import { useMemo, useState } from "react";

import type { PredictionResponse, SkeletonFrame } from "../lib/api";
import { requestPrediction } from "../lib/api";
import { sampleWindows, type SampleName } from "../lib/samples";

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type RequestState = "idle" | "loading" | "success" | "error";

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function SkeletonPreview({ frames }: { frames: SkeletonFrame[] }) {
  const latestFrame = frames[frames.length - 1];
  const points = Object.entries(latestFrame.keypoints);
  const lines = [
    ["head", "hip"],
    ["hip", "ankle"],
  ] as const;

  return (
    <svg aria-label="Latest skeleton frame" className="skeletonPreview" viewBox="0 0 100 100">
      <rect className="previewFloor" x="0" y="82" width="100" height="3" rx="1.5" />
      {lines.map(([from, to]) => {
        const start = latestFrame.keypoints[from];
        const end = latestFrame.keypoints[to];
        return (
          <line
            key={`${from}-${to}`}
            className="bone"
            x1={start.x * 100}
            y1={start.y * 100}
            x2={end.x * 100}
            y2={end.y * 100}
          />
        );
      })}
      {points.map(([name, point]) => (
        <circle
          key={name}
          className="joint"
          cx={point.x * 100}
          cy={point.y * 100}
          r={name === "head" ? 4.2 : 3.4}
        />
      ))}
    </svg>
  );
}

export default function Home() {
  const [sampleName, setSampleName] = useState<SampleName>("stable");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [requestState, setRequestState] = useState<RequestState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const selectedWindow = sampleWindows[sampleName];
  const isFall = prediction?.label === "fall";
  const statusLabel = prediction ? (isFall ? "Fall detected" : "Normal motion") : "Awaiting prediction";
  const statusIcon = prediction ? (
    isFall ? <Siren aria-hidden="true" /> : <CheckCircle2 aria-hidden="true" />
  ) : (
    <Activity aria-hidden="true" />
  );
  const probability = prediction?.fall_probability ?? 0;
  const confidence = prediction?.confidence ?? 0;
  const reasons = useMemo(() => prediction?.reasons ?? ["No prediction yet"], [prediction]);

  async function handlePredict() {
    setRequestState("loading");
    setErrorMessage(null);

    try {
      const result = await requestPrediction(apiBaseUrl, selectedWindow);
      setPrediction(result);
      setRequestState("success");
    } catch (error) {
      setPrediction(null);
      setRequestState("error");
      setErrorMessage(error instanceof Error ? error.message : "Prediction request failed");
    }
  }

  return (
    <main className="appShell">
      <section className="topBar" aria-label="System status">
        <div>
          <p className="eyebrow">Safe Guard</p>
          <h1>Monitoring Console</h1>
        </div>
        <div className={`statusBadge ${isFall ? "danger" : "normal"}`}>
          {statusIcon}
          <span>{statusLabel}</span>
        </div>
      </section>

      <section className="dashboardGrid">
        <div className="panel monitorPanel">
          <div className="panelHeader">
            <div>
              <p className="eyebrow">Skeleton Window</p>
              <h2>{selectedWindow.source_id}</h2>
            </div>
            <ShieldAlert aria-hidden="true" />
          </div>

          <SkeletonPreview frames={selectedWindow.frames} />

          <div className="segmentedControl" aria-label="Sample selector">
            {(["stable", "fall"] as SampleName[]).map((name) => (
              <button
                key={name}
                className={sampleName === name ? "active" : ""}
                type="button"
                onClick={() => {
                  setSampleName(name);
                  setPrediction(null);
                  setRequestState("idle");
                }}
              >
                {name === "stable" ? "Stable" : "Fall"}
              </button>
            ))}
          </div>

          <button className="primaryButton" type="button" onClick={handlePredict}>
            <Activity aria-hidden="true" />
            <span>{requestState === "loading" ? "Predicting" : "Run Prediction"}</span>
          </button>
        </div>

        <div className="panel resultPanel">
          <div className="panelHeader">
            <div>
              <p className="eyebrow">Prediction</p>
              <h2>{statusLabel}</h2>
            </div>
            <span className="modelName">{prediction?.model_name ?? "No model response"}</span>
          </div>

          <div className="metricGrid">
            <div>
              <span>Fall probability</span>
              <strong>{percent(probability)}</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>{percent(confidence)}</strong>
            </div>
          </div>

          <div className="meter" aria-label="Fall probability meter">
            <span style={{ width: percent(probability) }} />
          </div>

          <ul className="reasonList">
            {reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>

          {errorMessage ? <p className="errorText">{errorMessage}</p> : null}
        </div>
      </section>
    </main>
  );
}
