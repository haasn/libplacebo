# New release steps

## Pre-release (vX.Y.0-rcN)

1. Tag `vX.Y.0-rcN` on `master`

## Normal release (vX.Y.0)

1. Tag `vX.Y.0` on `master`
 - add a list of API additions (and changes/removals) at the very least
 - see e.g. `git show v6.338.0` for an example
 - the "fixes" section can be omitted if too lazy, but would be nice to have

2. Create version branch `vX.Y`
3. Force-push `release` branch (or fast-forward if possible)
4. Update topic on IRC #libplacebo
5. Bump 'X' version number in meson.build, for next release (optional)
  - if bumping major version number, anything with `PL_DEPRECATED_IN(v${X-2].0)`
    can/should be removed from public headers, where X is the new major version
    number.
  - Example: last v7 release (v7.Y.0) is tagged, and master gets new major
    version number *8*, so anything deprecated in v6.0 can be removed.
6. Tag release on github
  - copy/paste release notes from the tag (see step 1)

## Bugfix release (vX.Y.Z)

1. Cherry-pick bug fixes onto version branch (`vX.Y`)
2. Update `Z` version number in `meson.build`
3. Tag `vX.Y.Z` on this branch
  - include list of fixes only since last bugfix release
4. Fast-forward `release` branch iff this is the latest major release
5. Update topic on IRC #libplacebo
6. Tag release on github
  - copy/paste release notes from the tag (see step 3)
