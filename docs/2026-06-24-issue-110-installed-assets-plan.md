# Plan: Issue #110 - Installed variants show needs download

**Issue:** https://github.com/graywzc/nvidia-gddr6-fan-control/issues/110
**Branch:** codex/issue-110-installed-assets
**Status:** implemented

## Requirements

- Variants with model assets already present on disk should not display
  `needs download` after the observer restarts.
- The running variant should continue to be marked installed.
- Observer-driven installs should continue to update the dashboard immediately.

## Non-Goals

- Do not change club-3090 setup or switch behavior.
- Do not perform expensive filesystem scans during every SSE snapshot render.

## Proposed Changes

- Detect installed assets from the club-3090 model cache during catalog refresh.
- Merge disk-detected variants into the existing installed-assets map without
  overwriting live install/run metadata.
- Add unit tests for model-cache and profile/weight-key detection.

## Verification

- `python3 -m unittest tests.test_observer`

## Risks

- Cache layouts can vary. The detector checks the model cache root and common
  profile/weight-key subdirectories, and only marks paths that contain files.

## Progress

- [x] Plan authored
- [x] Implementation
- [x] Tests
- [x] Lead review
- [ ] PR merged / issue closed
