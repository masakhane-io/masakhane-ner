#!/usr/bin/env python3
"""Read-only structural and split-overlap audit for MasakhaNER CoNLL data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

TAG = re.compile(r'^(O|[BI]-.+)$')
SPLITS = ('train', 'dev', 'test')


def audit_corpus(data_dir, max_examples=20):
    """Count later exact-token occurrences first seen in another split.

    Matching is within a language, preserves accents/case, and does not certify
    annotations. Examples are bounded; all flags are counted. No data is edited.
    """
    root = Path(data_dir)
    if max_examples < 0:
        raise ValueError('max_examples must be nonnegative')
    files = sorted(p for split in SPLITS for p in root.glob('*/' + split + '.txt'))
    if not files:
        raise ValueError('No language/{train,dev,test}.txt files found')
    counts, severities = Counter(), Counter()
    examples, inventory = [], []

    def flag(kind, severity, path, line, **details):
        counts[kind] += 1
        severities[severity] += 1
        if len(examples) < max_examples:
            examples.append(dict(kind=kind, severity=severity,
                file=path.relative_to(root).as_posix(), line=line, **details))

    for lang in sorted({p.parent for p in files}):
        seen = {}
        for split in SPLITS:
            path = lang / (split + '.txt')
            if path not in files:
                continue
            words, labels, previous, sentence_line = [], [], 'O', 0
            sentences = tokens = 0

            def finish():
                nonlocal sentences
                if not words:
                    return
                sentences += 1
                key = hashlib.sha256(json.dumps(words, ensure_ascii=False).encode('utf-8')).hexdigest()
                location = dict(file=path.relative_to(root).as_posix(), line=sentence_line, split=split)
                if key in seen and seen[key]['location']['split'] != split:
                    first = seen[key]
                    flag('cross_split_repeat', 'warning', path, sentence_line,
                         first=first['location'], token_count=len(words), fingerprint=key)
                    if first['labels'] != labels:
                        flag('conflicting_labels', 'warning', path, sentence_line,
                             first=first['location'], fingerprint=key)
                elif key not in seen:
                    seen[key] = dict(location=location, labels=list(labels))
            try:
                with path.open(encoding='utf-8') as stream:
                    for number, line in enumerate(stream, 1):
                        fields = line.split()
                        if not fields or fields[0] == '-DOCSTART-':
                            finish()
                            words, labels, previous = [], [], 'O'
                            continue
                        if not words:
                            sentence_line = number
                        if len(fields) != 2:
                            flag('column_count', 'error', path, number, columns=len(fields))
                        word, tag = fields[0], fields[-1] if len(fields) > 1 else ''
                        words.append(word)
                        labels.append(tag)
                        tokens += 1
                        if not TAG.fullmatch(tag):
                            flag('invalid_tag', 'error', path, number, tag=tag)
                        elif tag.startswith('I-') and previous not in ('B-' + tag[2:], 'I-' + tag[2:]):
                            flag('iob2_transition', 'warning', path, number, previous=previous, tag=tag)
                        if '\ufffd' in word:
                            flag('replacement_character', 'warning', path, number)
                        previous = tag
                finish()
            except (OSError, UnicodeError) as exc:
                flag('read_error', 'error', path, 0, message=str(exc))
            inventory.append(dict(language=lang.name, split=split,
                file=path.relative_to(root).as_posix(), sentences=sentences, tokens=tokens))
    return dict(schema_version=1, languages=len({p.parent.name for p in files}), files=inventory,
        sentences=sum(row['sentences'] for row in inventory), tokens=sum(row['tokens'] for row in inventory),
        counts=dict(sorted(counts.items())), severity_counts=dict(severities), examples=examples,
        omitted_examples=sum(counts.values()) - len(examples),
        note='Warnings require review; repetitions and IOB2 transitions are not automatically annotation errors.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-examples', type=int, default=20)
    parser.add_argument('--fail-on', choices=('error', 'warning', 'never'), default='error')
    args = parser.parse_args(argv)
    try:
        inputs = {p.resolve() for split in SPLITS for p in args.data_dir.glob('*/' + split + '.txt')}
        if args.output and args.output.resolve() in inputs:
            raise ValueError('Output must not overwrite a corpus input file')
        report = audit_corpus(args.data_dir, args.max_examples)
        rendered = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
        if args.output:
            args.output.write_text(rendered, encoding='utf-8')
        else:
            print(rendered, end='')
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    counts = report['severity_counts']
    failed = counts.get('error', 0) or (args.fail_on == 'warning' and counts.get('warning', 0))
    return int(args.fail_on != 'never' and bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
