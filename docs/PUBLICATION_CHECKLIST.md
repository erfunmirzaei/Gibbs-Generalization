# Public Release Checklist

Use this checklist before switching the GitHub repository to public.

- Update the README citation with the official ICML title, author list, proceedings entry, and DOI or OpenReview URL if available.
- Decide whether all tracked CSV and plot artifacts should remain in the public repository, or whether a smaller curated artifact bundle should be linked from the README.
- Confirm that no local datasets, private paths, API keys, or machine-specific Conda environments are tracked.
- Run a smoke test with `TEST_MODE = True` for `main.py` and `master.py`.
- Regenerate at least one representative plot from a tracked CSV.
- Tag the release commit, for example `v1.0-icml`.
- Add paper-to-artifact mapping notes if reviewers/readers need to reproduce specific figures or tables.
