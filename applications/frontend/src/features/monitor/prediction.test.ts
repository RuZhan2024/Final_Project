import { extractPredictionState } from "./prediction";

describe("monitor prediction state parsing", () => {
  const baseArgs = {
    mode: "tcn",
    previousStable: { fall: 0, uncertain: 0, safe: 0, last: "not_fall" },
    settingsPayload: null,
  };

  it("keeps live triage smoothing enabled by default", () => {
    const first = extractPredictionState({
      ...baseArgs,
      data: {
        triage_state: "fall",
        safe_alert: true,
        models: { tcn: { mu: 0.92 } },
      },
    });

    expect(first.triageState).toBe("not_fall");

    const second = extractPredictionState({
      ...baseArgs,
      previousStable: first.stable,
      data: {
        triage_state: "fall",
        safe_alert: true,
        models: { tcn: { mu: 0.91 } },
      },
    });

    expect(second.triageState).toBe("fall");
  });

  it("trusts the backend triage state immediately when smoothing is disabled", () => {
    const parsed = extractPredictionState({
      ...baseArgs,
      smoothTriage: false,
      data: {
        triage_state: "fall",
        safe_alert: true,
        models: { tcn: { mu: 0.93 } },
      },
    });

    expect(parsed.triageState).toBe("fall");
    expect(parsed.markerKind).toBe("fall");
  });

  it("prefers canonical triage_state over legacy safe_state", () => {
    const parsed = extractPredictionState({
      ...baseArgs,
      smoothTriage: false,
      data: {
        triage_state: "fall",
        safe_state: "not_fall",
        safe_alert: true,
        models: { tcn: { mu: 0.94 } },
      },
    });

    expect(parsed.triageState).toBe("fall");
  });
});
