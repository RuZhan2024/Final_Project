# Front-end (skeleton extraction + mode switch)

Your front-end (e.g. React) is responsible for:

1) Capturing the RGB stream.
2) Extracting skeleton keypoints per frame (e.g. MediaPipe Pose / BlazePose).
3) Building a sliding window (e.g. **W=48**, **S=12**) of `(xy, conf)`.
4) Sending each window to the server endpoint:

```
POST /api/monitor/predict_window
```

5) Switching `mode` on the UI:
- `tcn`
- `gcn`
- `dual` (TCN+GCN)

Payload format is documented in `server/README.md`.

Tip: send `timestamp_ms` from the capture clock so server-side state machines stay stable.
