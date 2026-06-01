import { extractPredictionState } from "./prediction";

describe("monitor prediction state parsing", () => {
  const baseArgs = {
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

  it("uses triage_state as the canonical visible state", () => {
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

  it("displays backend policy score before raw model probability", () => {
    const parsed = extractPredictionState({
      ...baseArgs,
      smoothTriage: false,
      data: {
        triage_state: "not_fall",
        safe_alert: false,
        models: {
          tcn: {
            policy_score: 0.34,
            mu: 0.95,
            triage: { ps: 0.95, tau_high: 0.41 },
          },
        },
      },
    });

    expect(parsed.triageState).toBe("not_fall");
    expect(parsed.pFall).toBeCloseTo(0.34, 6);
    expect(parsed.markerKind).toBe("safe");
  });

  it("falls back to the tracker score when no policy score is supplied", () => {
    const parsed = extractPredictionState({
      ...baseArgs,
      smoothTriage: false,
      data: {
        triage_state: "uncertain",
        safe_alert: false,
        models: {
          tcn: {
            triage: { ps: 0.37, tau_high: 0.41 },
          },
        },
      },
    });

    expect(parsed.triageState).toBe("uncertain");
    expect(parsed.pFall).toBeCloseTo(0.37, 6);
  });
});
