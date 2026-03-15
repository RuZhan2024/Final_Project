# AGENTS.md

## Mission
This repository is a real fall detection project with dataset-specific preprocessing, labeling, subject-safe splits, window generation, training, threshold fitting, evaluation, and deployment logic.

Your job is to transform this repository into a premium instructor-led course that teaches students how to rebuild the project from zero.

## Delivery model
The teaching documents are the primary output.

Hard rules:
- Do NOT create a separate `teaching/` source directory.
- Do NOT create parallel teaching implementation files as standalone code files.
- All teaching code must appear directly inside markdown lesson documents as code blocks.
- The student must be able to rebuild the teaching project by copying code from the lesson notes.

## Teaching style
Write like a premium Udemy-style course:
- patient
- explicit
- implementation-first
- no skipped steps
- no "implement this yourself"
- no "boilerplate omitted"
- no shallow summaries where teaching is required

For every daily lesson:
1. state today's goal
2. explain why this part exists in the overall pipeline
3. show the file tree snapshot for today
4. include the full code for all files introduced or changed today
5. explain the code line by line or block by block
6. show exact run commands
7. show expected outputs
8. show sanity checks
9. list common bugs and fixes
10. map the lesson back to the original repository
11. preview the next lesson

## Code explanation rules
- For files under 120 lines: explain line by line.
- For longer files: explain section by section, and explain all non-trivial lines individually.
- For every important function, explain:
  - inputs
  - outputs
  - side effects
  - why it exists in the pipeline
- When introducing a new concept, give a tiny concrete example first.

## Pacing rules
- Split each week into 7 daily lessons.
- Each day should introduce only a small number of concepts.
- Prefer 1 to 3 files per day.
- Prefer one clear runnable milestone per day.
- The student should be able to complete one day in a focused study session.

## Architecture rules
- Explain the system in execution order, not folder order.
- Keep the teaching path aligned with the real repository architecture.
- Distinguish clearly between:
  1. teaching simplification
  2. intermediate engineering version
  3. full research/project version
- Exclude dead code, legacy experiments, and non-essential complexity from early lessons unless needed later.

## Verification
When possible:
- trace actual Makefile targets, configs, scripts, and imports
- verify claims against the real repository
- run lightweight checks
- confirm commands and outputs where feasible

You may use temporary scratch files internally to validate examples, but the final deliverables must be markdown documents only.
Do not leave a permanent teaching code directory in the repository.

## Output layout
Write outputs under:

docs/course/week01/
docs/course/week02/
docs/course/week03/
docs/course/week04/
docs/course/week05/

Each week must contain:
- README.md
- day01.md
- day02.md
- day03.md
- day04.md
- day05.md
- day06.md
- day07.md