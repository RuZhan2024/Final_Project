# Frontend

This directory contains the React frontend for Safe Guard Fall Detection.

For the main project overview, recommended demo path, deployment link, and
evidence links, see the repository root `README.md`.

## Requirements

- Node.js / npm
- Node `22.22.x` recommended for parity with the current frontend build
- backend API running locally or configured through `REACT_APP_API_BASE`

## Recommended Run Path

From the repository root:

```bash
make bootstrap-dev
```

This starts both the backend and frontend for the local demo workflow.

Open:

- frontend: `http://localhost:3000`
- backend health: `http://127.0.0.1:8000/api/health`

## Frontend-Only Development

Use this path when the backend is already running.

```bash
cd applications/frontend
npm install
npm start
```

By default, the frontend expects the backend at `http://localhost:8000`.

## Environment

To point the frontend at a non-default backend, create
`applications/frontend/.env`:

```bash
REACT_APP_API_BASE=http://localhost:8000
```

`REACT_APP_API_BASE` is read by `applications/frontend/src/lib/config.ts`.

## Build and Type Check

```bash
cd applications/frontend
npm run typecheck
npm run build
```

The repository root also provides a clean-install frontend parity check:

```bash
make frontend-render-check
```

This runs:

- `npm ci`
- `npm run build`

## Replay Clips

The `Monitor` replay dropdown is populated by the backend from the configured
replay clips directory.

Default path:

- `ops/deploy_assets/replay_clips`

Override:

```bash
REPLAY_CLIPS_DIR=/absolute/path/to/replay_clips
```

Supported replay extensions:

- `.mp4`
- `.mov`
- `.webm`
- `.m4v`

Quick check:

```bash
curl -s http://localhost:8000/api/replay/clips
```

If this returns an empty `clips` list, the replay selector in the frontend will
also be empty.

## Useful API Checks

Health:

```bash
curl -s http://localhost:8000/api/health
```

Runtime specs:

```bash
curl -s http://localhost:8000/api/spec
```

## Notes

- default frontend port: `3000`
- default backend port: `8000`
- the UI supports backend `/api/*` and `/api/v1/*` compatibility routes
- for CORS changes, use the backend configuration documented in the root README and `.env.example`
