# Masakhane Analysis Scripts

This directory contains some scripts to run [interpretable evaluation](https://github.com/neulab/InterpretEval) on the Masakhane data.

Check out `run_all_analyses.sh` to get started.

When it is done running, you should have analysis reports included in `InterpretEval/interpretEval/analysis/tEval-ner-masakhane-*.html`

## Corpus integrity audit (no model dependencies)

From the repository root:

```bash
python analysis_scripts/audit_corpus.py --data-dir MasakhaNER2.0/data --output audit.json
python -m pytest -q tests/test_audit_corpus.py
```

Reports file/line locations for malformed rows, invalid BIO tags, IOB2
transition flags, Unicode replacement characters, exact token sequences
repeated across splits of the same language, and conflicting labels on
repeats. It never edits data or normalizes accents. Warnings require review:
repeats and IOB2 transitions are not automatically annotation errors. Exact
matching does not detect paraphrases or all forms of contamination. Counts
cover every finding; `--max-examples` bounds only the example list. Missing
splits are allowed. No input files or output-write failures return 2;
structural findings and per-file read failures return 1; warnings return 0 unless `--fail-on warning` is selected.
`--fail-on never` reports findings without failing for their severity.
