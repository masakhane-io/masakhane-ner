# coding=utf-8
# David Adelani

'''

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("Davlan/distilbert-base-multilingual-cased-masakhaner")
model = AutoModelForTokenClassification.from_pretrained("Davlan/distilbert-base-multilingual-cased-masakhaner")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Emir of Kano turban Zhang wey don spend 18 years for Nigeria"
ner_results = nlp(example)
print(ner_results)
'''
import logging
import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss
import torch
import string
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

def get_labels():
    return ["O", "B-DATE", "I-DATE", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples(text):
    guid_index = 1
    examples = []

    sentences = text.splitlines()
    mode = 'test'
    for sent in sentences:
        sent = sent.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
        words = sent.split()
        labels = ['O']*len(words)
        if words:
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
            guid_index += 1

    return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        #print(ex_index, len(example.words))
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        try:
            assert len(label_ids) == max_seq_length
        except:
            continue

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
        )
    return features


def load_and_cache_examples(text, tokenizer, labels, pad_token_label_id, max_seq_length, model_type):
    examples = read_examples(text)
    features = convert_examples_to_features(
        examples,
        labels,
        max_seq_length,
        tokenizer,
        cls_token_at_end=bool(model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
        pad_token_label_id=pad_token_label_id,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def evaluate(model, tokenizer, labels, pad_token_label_id, text, max_seq_length, model_type, device):

    #load_and_cache_examples(text, tokenizer, labels, pad_token_label_id, max_seq_length, model_type)
    eval_dataset = load_and_cache_examples(text, tokenizer, labels, pad_token_label_id, max_seq_length, model_type)

    eval_batch_size = 16 #args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) #if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    '''
    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        'report': classification_report(out_label_list, preds_list),
    }
    '''

    #for key in sorted(results.keys()):
    #    logger.info("  %s = %s", key, str(results[key]))

    return preds_list


def predict_tags(text, model_name_or_path, model_type):
    # Prepare CONLL-2003 task
    labels = get_labels()
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    model_name_or_path = "Davlan/distilbert-base-multilingual-cased-masakhaner"
    config_class, model_class, tokenizer_class = AutoConfig, AutoModelForTokenClassification, AutoTokenizer  # MODEL_CLASSES[args.model_type]

    '''
    config = config_class.from_pretrained(model_name_or_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=None,
    )
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
        cache_dir=None,
    )
    model = model_class.from_pretrained(model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=None,
    )
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # Evaluation
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=False)
    model = model_class.from_pretrained(model_name_or_path)
    model.to(device)
    max_seq_length = 200
    model_type = "distilbert"
    predictions = evaluate(model, tokenizer, labels, pad_token_label_id, text, max_seq_length, model_type, device)
    # evaluate(model, tokenizer, labels, pad_token_label_id, mode="test")
    # Save predictions

    sentences = text.splitlines()
    output_test_predictions_file = 'test_predictions.txt'
    with open(output_test_predictions_file, "w") as writer:
        for example_id, sent in enumerate(sentences):
            sent = sent.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
            words = sent.split()

            for word in words:
                label = predictions[example_id].pop(0)
                writer.write(word + " " + label + "\n")
                print(word + " " + label + "\n")
            writer.write("\n")
            print("\n")


if __name__ == "__main__":
    text = "Emir of Kano turban Zhang wey don spend 18 years for Nigeria"
    model_name = "Davlan/distilbert-base-multilingual-cased-masakhaner"
    model_type = 'distilbert'
    predict_tags(text, model_name, model_type)
