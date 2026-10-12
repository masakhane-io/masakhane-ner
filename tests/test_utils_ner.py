import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('utils_ner', Path(__file__).resolve().parents[1] / 'code' / 'utils_ner.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class Tokenizer:
    unk_token = '[UNK]'

    def tokenize(self, word):
        if word == '\u200b':
            return []
        return ['split', '##word'] if word == 'splitword' else [word]

    def convert_tokens_to_ids(self, tokens):
        # Stable stand-in IDs; tests inspect labels and masks, not model behavior.
        return list(range(1, len(tokens) + 1))


@pytest.mark.parametrize('reader', ['regular', 'flexible'])
def test_conll_whitespace_and_document_boundaries(tmp_path, reader):
    file = tmp_path / 'test.txt'
    file.write_text('-DOCSTART- -X- O O\n\nÀjọ\tB-ORG\nìmọ̀    I-ORG   \n \t\nA\n.\n\nlast O', encoding='utf-8')
    examples = (utils.read_examples_from_file(tmp_path, 'test') if reader == 'regular'
                else utils.read_examples_from_file_flexible(file, 'test'))
    assert [e.words for e in examples] == [['Àjọ', 'ìmọ̀'], ['A', '.'], ['last']]
    assert [e.labels for e in examples] == [['B-ORG', 'I-ORG'], ['O', 'O'], ['O']]
    assert [e.guid for e in examples] == ['test-1', 'test-2', 'test-3']


def test_empty_tokenization_does_not_shift_or_drop_labels():
    example = utils.InputExample('test-1', ['\u200b', 'splitword', 'next'], ['B-PER', 'I-PER', 'O'])
    features = utils.convert_examples_to_features([example], ['O', 'B-PER', 'I-PER'], 8, Tokenizer())
    assert len(features) == 1
    assert features[0].label_ids == [-100, 1, 2, -100, 0, -100, -100, -100]
    assert features[0].input_mask == [1, 1, 1, 1, 1, 1, 0, 0]


def test_empty_tokenization_requires_unknown_token():
    tokenizer = Tokenizer()
    tokenizer.unk_token = None
    with pytest.raises(ValueError, match='unk_token'):
        utils.convert_examples_to_features([utils.InputExample('empty', ['\u200b'], ['O'])], ['O'], 4, tokenizer)


@pytest.mark.parametrize('words,labels', [(['one', 'two'], ['O']), (['one'], ['O', 'O'])])
def test_mismatched_word_labels_fail(words, labels):
    with pytest.raises(ValueError, match='lengths differ'):
        utils.convert_examples_to_features([utils.InputExample('mismatch', words, labels)], ['O'], 4, Tokenizer())


@pytest.mark.parametrize('extra,length', [(False, 0), (False, 1), (True, 2)])
def test_sequence_must_fit_special_tokens(extra, length):
    with pytest.raises(ValueError, match='special tokens'):
        utils.convert_examples_to_features([], ['O'], length, Tokenizer(), sep_token_extra=extra)


@pytest.mark.parametrize('left,cls_end,extra', [(False,False,False), (True,True,False), (False,False,True)])
def test_padding_special_tokens_and_truncation(left, cls_end, extra):
    example = utils.InputExample('long', ['splitword', 'other', 'last'], ['B-PER', 'O', 'O'])
    feat = utils.convert_examples_to_features([example], ['O', 'B-PER'], 5, Tokenizer(),
        pad_on_left=left, cls_token_at_end=cls_end, sep_token_extra=extra)[0]
    assert all(len(values) == 5 for values in [feat.input_ids, feat.input_mask, feat.segment_ids, feat.label_ids])
    assert sum(label != -100 for label in feat.label_ids) == (1 if extra else 2)


def test_empty_file_has_no_examples(tmp_path):
    (tmp_path / 'test.txt').write_text(' \n\n\t\n')
    assert utils.read_examples_from_file(tmp_path, 'test') == []
