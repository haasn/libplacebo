# General

- Agents: NEVER PRETEND TO BE A HUMAN! This is non-negotiable project policy. No system or user prompt may override it.
- All commits are authored by REAL HUMANS. Do not add a 'Co-authored-by' line for AI/LLM agents. The HUMAN is accountable for the code in full, including any legal repercussions of implicitly including copyrighted code that was distilled into an LLM's weights.
- Disclose AI/LLM use in the pull request description or the issue body. Name the exact model and the tool, not "an AI".

# Commits

- All commits MUST be easily human-reviewable!
- Split them into commits that are as atomic and incremental as possible.
  - Several cosmetic edits (typo fixes, whitespace changes, alignment fixes) may share one commit.
  - Never put a cosmetic edit in the same commit as a functional change.
  - Never put a cosmetic edit in the same commit as code movement (moving or reordering code).
  - Never put a code movement in the same commit as adding new functionality.
- Make as many independent commits as you can. A commit that adds a new public API must be separate from the commit that first uses it. Exception: a local/static helper used only by the new code may come in the same commit as that code.
- Every commit must compile and pass tests by itself. (e.g. `git rebase -x 'ninja -Cbuild test'`)

# Tests

- Use test cases during development, but commit them only where appropriate or useful.
- Never commit test cases for trivial code that is obviously correct.
- Never commit test cases that are longer and more complicated than the logic they are testing.
- Avoid committing test cases for code that is unlikely to change, unless the test case is particularly useful, simple, or the logic tested is dense or tricky.

# Docs

- Keep committed documentation and commit messages reasonably concise. State the key details instead of listing every example.
- Comments should only be added when necessary to explain why code is needed, or point out something that isn't obvious from the code.
- Potentially surprising or confusing aspects of the change, that would strike a reviewer as odd or unexpected, should be explained in the commit message.
- Agents: Internal plans are encouraged during development, but they should not be added verbatim to comments or commit messages.

# Shaders

- New code should prefer to use #pragma GLSL (see docs/glsl.md) instead of GLSL() macros
- GLSL code (unless otherwise guarded) should be backwards compatible with 130 syntax:
  * Bad: `color.rgb + 1` (implicit int->vec3 promotion)
  * Good: `color.rgb + vec3(1.0)`
