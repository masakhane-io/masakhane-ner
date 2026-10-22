import importlib.util
from pathlib import Path
import json
import pytest

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'analysis_scripts/audit_corpus.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write(root, lang, split, text):
    folder = root / lang
    folder.mkdir(exist_ok=True)
    (folder / (split + '.txt')).write_text(text, encoding='utf-8')


def test_cross_split_exact_tokens_and_locations(tmp_path):
    write(tmp_path, 'yor', 'train', 'Àjọ B-ORG\nìmọ̀ I-ORG\n\n')
    write(tmp_path, 'yor', 'test', 'Àjọ B-ORG\nìmọ̀ I-ORG')
    original = {p: p.read_bytes() for p in tmp_path.rglob('*.txt')}
    report = audit.audit_corpus(tmp_path)
    assert report['counts'] == {'cross_split_repeat': 1}
    assert report['sentences'] == 2 and report['tokens'] == 4
    assert report['examples'][0]['first']['split'] == 'train'
    assert report['examples'][0]['line'] == 1
    assert all(p.read_bytes() == data for p, data in original.items())


def test_no_cross_language_or_accent_collapse(tmp_path):
    write(tmp_path, 'yor', 'train', 'ọ B-LOC\n')
    write(tmp_path, 'yor', 'test', 'o B-LOC\n')
    write(tmp_path, 'ibo', 'test', 'ọ B-LOC\n')
    assert audit.audit_corpus(tmp_path)['counts'] == {}


def test_structural_and_annotation_flags_are_separate(tmp_path):
    write(tmp_path, 'yor', 'train', 'one I-PER\ntwo I-ORG\nthree INVALID\nfour\nfive O extra\n� O\n')
    report = audit.audit_corpus(tmp_path, max_examples=1)
    assert report['counts']['iob2_transition'] == 2
    assert report['counts']['column_count'] == 2
    assert report['counts']['replacement_character'] == 1
    assert report['severity_counts']['error'] > 0
    assert len(report['examples']) == 1
    assert report['omitted_examples'] == sum(report['counts'].values()) - 1


def test_document_boundary_resets_bio(tmp_path):
    write(tmp_path, 'yor', 'train', 'one B-PER\n-DOCSTART- -X- O O\ntwo I-PER\n')
    report = audit.audit_corpus(tmp_path)
    assert report['sentences'] == 2
    assert report['counts']['iob2_transition'] == 1


def test_conflicting_labels_are_reported_not_repaired(tmp_path):
    write(tmp_path, 'yor', 'train', 'one B-PER\n')
    write(tmp_path, 'yor', 'dev', 'one B-ORG\n')
    assert audit.audit_corpus(tmp_path)['counts']['conflicting_labels'] == 1


def test_cli_and_failure_policy(tmp_path, capsys):
    write(tmp_path, 'yor', 'train', 'one I-PER\n')
    assert audit.main(['--data-dir', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)['sentences'] == 1
    assert audit.main(['--data-dir', str(tmp_path), '--fail-on', 'warning']) == 1
    assert audit.main(['--data-dir', str(tmp_path), '--max-examples', '-1']) == 2
    source = tmp_path / 'yor/train.txt'
    assert audit.main(['--data-dir', str(tmp_path), '--output', str(source)]) == 2
    assert source.read_text() == 'one I-PER\n'


def test_no_files(tmp_path):
    with pytest.raises(ValueError, match='No language'):
        audit.audit_corpus(tmp_path)
