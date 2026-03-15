# Supervisor Delivery Modes

This project supports two practical delivery modes for supervisor review.

Use this page when deciding how much setup the reviewer should be asked to do.

## Mode 1: Lightweight Demo Mode

Recommended when the goal is to review:

- frontend/backend integration
- live or replay inference
- monitor UI behavior
- fall detection output

Start command:

```bash
make bootstrap-dev
```

Characteristics:

- no MySQL setup required
- no DB env vars required
- many DB-related API paths fall back gracefully
- lowest setup friction for a reviewer

Best for:

- demos
- meetings
- quick validation on another machine

## Mode 2: Full Persistent-System Mode

Recommended when the goal is to review:

- event persistence
- settings persistence
- caregiver/event database flows
- a more complete deployed-system shape

Preferred delivery approach:

- `docker compose`

Start command:

```bash
docker compose up
```

If host port `3306` is already in use:

```bash
MYSQL_PORT=3307 docker compose up
```

Rationale:

- frontend + backend + MySQL can be started together
- MySQL can use a persistent volume
- lower reviewer setup cost than asking them to install and configure MySQL manually

If Docker is not provided, the manual fallback is:

- install MySQL
- create/import the project database
- set:
  - `DB_HOST`
  - `DB_PORT`
  - `DB_USER`
  - `DB_PASS`
  - `DB_NAME`

## Recommendation

For supervisor delivery, keep both options available:

1. lightweight no-DB mode for the fastest review
2. Docker-backed persistent mode for the full-system review

This avoids forcing the reviewer to solve infrastructure setup before they can see the project working.
