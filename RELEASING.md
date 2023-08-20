# New release steps

## Pre-release (vX.Y.0-rcN)

1. Tag `vX.Y.0-rcN` on `master`

## Normal release (vX.Y.0)

1. Tag `vX.Y.0` on `master`
2. Create version branch `vX.Y`
3. Force-push `release` branch (or fast-forward if possible)
4. Update topic on IRC #libplacebo
5. Bump 'X' version number in meson.build, for next release (optional)
6. Tag release on github

## Bugfix release (vX.Y.Z)

1. Cherry-pick bug fixes onto version branch (`vX.Y`)
2. Update `Z` version number in `meson.build`
3. Tag `vX.Y.Z` on this branch
4. Fast-forward `release` branch iff this is the latest major release
5. Update topic on IRC #libplacebo
6. Tag release on github
