# New release steps

## Pre-release (vX.Y.0-rcN)

1. Tag `vX.Y.0-rcN` on `master`

## Major release (vX.Y.0)

1. Tag `vX.Y.0` on `master`
2. Create version branch `vX.Y`
3. Force-push `release` branch (or fast-forward if possible)

## Bugfix release (vX.Y.Z)

1. Cherry-pick bug fixes onto version branch (`vX.Y`)
2. Update `Z` version number in `meson.build`
3. Tag `vX.Y.Z` on this branch
4. Fast-forward `release` branch iff this is the latest major release
