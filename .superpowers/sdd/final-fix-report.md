# Final whole-branch review fixes

Date: 2026-10-04
Branch: `feat/third-eye-voice-mvp`

## Critical 1 — repository safeguards

- Added and committed `.gitignore`, including `.env`, local Python environments,
  model/cache output, editor files, and macOS metadata.
- Removed the tracked `.DS_Store` from the Git index with
  `git rm --cached .DS_Store`; the local file remains ignored.

## Critical 2 — audible voice failures

- Deepgram session connection/run exceptions now end and disarm the session,
  return it to `IDLE`, and say: “Speech is down, try again.”
- `AgentV1Error` messages now end and disarm the session. OpenRouter/LLM/think
  failures say: “I can't think right now.” Other agent errors use the speech
  failure phrase.
- A per-session guard prevents repeated error messages from spamming the user.
- Added focused tests for run-loop failures and `AgentV1Error` handling.

## Critical 3 — stale SceneStore meters

- Added `SCENE_STALE_MS = 1500`.
- `sense_snapshot`, `measure_distances`, `find_object`, and `describe_scene`
  now reject frames older than the threshold with
  `{"ok": false, "error": "No current camera frame"}`.
- Added a parameterized regression test covering all four tools. The test was
  observed failing for every tool before the freshness check was implemented.

## Important 8 — dependency compatibility

- Pinned `deepgram-sdk>=7,<8` in `requirements.txt`.
- Verification environment uses `deepgram-sdk` 7.12.0.

## Verification

Command:

` .venv/bin/pytest tests/ -q `

Result:

`40 passed in 0.33s`

`git diff --check` also completed without errors before final staging.

No OCR or transit work was included.
