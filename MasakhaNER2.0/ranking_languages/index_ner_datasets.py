import os
import subprocess
import numpy as np
from collections import Counter


def read_data(fn):
	with open(fn) as inp:
		lines = inp.readlines()
	c = []
	v = []
	for l in lines:
		l = l.strip().split()
		if len(l) == 2:
			c.append(int(l[1]))
			v.append(l[0])
	return v,c


dataset_dir = "ranking_data/datasets/"

# Not needed now
#eng_vocab_f = "datasets/eng/word.vocab"
#en_v, en_c = read_data(eng_vocab_f)
#eng_vocab_f = "datasets/eng/subword.vocab"
#en_sub_v, en_sub_c = read_data(eng_vocab_f)

#w2i = {w:i for i,w in enumerate(en_v)}
#subw2i = {w:i for i,w in enumerate(en_sub_v)}

def get_vocab(filename):
	with open(filename, encoding='latin-1') as inp:
		lines = inp.readlines()
	all_words = [w for l in lines for w in l.strip().split()]
	return all_words

#LETTER_CODES = {"am": "amh", "bs": "bos", "vi": "vie", "wa": "wln", "eu": "eus", "so": "som", "el": "ell", "aa": "aar", "or": "ori", "sm": "smo", "gn": "grn", "mi": "mri", "pi": "pli", "ps": "pus", "ms": "msa", "sa": "san", "ko": "kor", "sd": "snd", "hz": "her", "ks": "kas", "fo": "fao", "iu": "iku", "tg": "tgk", "dz": "dzo", "ar": "ara", "fa": "fas", "es": "spa", "my": "mya", "mg": "mlg", "st": "sot", "gu": "guj", "uk": "ukr", "lv": "lav", "to": "ton", "nv": "nav", "kl": "kal", "ka": "kat", "yi": "yid", "pl": "pol", "ht": "hat", "lu": "lub", "fr": "fra", "ia": "ina", "lt": "lit", "om": "orm", "qu": "que", "no": "nor", "sr": "srp", "br": "bre", "rm": "roh", "io": "ido", "gl": "glg", "nb": "nob", "ng": "ndo", "ts": "tso", "nr": "nbl", "ee": "ewe", "bo": "bod", "mt": "mlt", "ta": "tam", "et": "est", "yo": "yor", "tw": "twi", "sl": "slv", "su": "sun", "gv": "glv", "lo": "lao", "af": "afr", "sg": "sag", "sv": "swe", "ne": "nep", "ie": "ile", "bm": "bam", "sc": "srd", "sw": "swa", "nn": "nno", "ho": "hmo", "ak": "aka", "ab": "abk", "ti": "tir", "fy": "fry", "cr": "cre", "sh": "hbs", "ny": "nya", "uz": "uzb", "as": "asm", "ky": "kir", "av": "ava", "ig": "ibo", "zh": "zho", "tr": "tur", "hu": "hun", "pt": "por", "fj": "fij", "hr": "hrv", "it": "ita", "te": "tel", "rw": "kin", "kk": "kaz", "hy": "hye", "wo": "wol", "jv": "jav", "oc": "oci", "kn": "kan", "cu": "chu", "ln": "lin", "ha": "hau", "ru": "rus", "pa": "pan", "cv": "chv", "ss": "ssw", "ki": "kik", "ga": "gle", "dv": "div", "vo": "vol", "lb": "ltz", "ce": "che", "oj": "oji", "th": "tha", "ff": "ful", "kv": "kom", "tk": "tuk", "kr": "kau", "bg": "bul", "tt": "tat", "ml": "mal", "tl": "tgl", "mr": "mar", "hi": "hin", "ku": "kur", "na": "nau", "li": "lim", "nl": "nld", "nd": "nde", "os": "oss", "la": "lat", "bn": "ben", "kw": "cor", "id": "ind", "ay": "aym", "xh": "xho", "zu": "zul", "cs": "ces", "sn": "sna", "de": "deu", "co": "cos", "sk": "slk", "ug": "uig", "rn": "run", "he": "heb", "ba": "bak", "ro": "ron", "be": "bel", "ca": "cat", "kj": "kua", "ja": "jpn", "ch": "cha", "ik": "ipk", "bi": "bis", "an": "arg", "cy": "cym", "tn": "tsn", "mk": "mkd", "ve": "ven", "eo": "epo", "kg": "kon", "km": "khm", "se": "sme", "ii": "iii", "az": "aze", "en": "eng", "ur": "urd", "za": "zha", "is": "isl", "mh": "mah", "mn": "mon", "sq": "sqi", "lg": "lug", "gd": "gla", "fi": "fin", "ty": "tah", "da": "dan", "si": "sin", "ae": "ave", "alb": "sqi", "arm": "hye", "baq": "eus", "tib": "bod", "bur": "mya", "cze": "ces", "chi": "zho", "wel": "cym", "ger": "deu", "dut": "nld", "gre": "ell", "per": "fas", "fre": "fra", "geo": "kat", "ice": "isl", "mac": "mkd", "mao": "mri", "may": "msa", "rum": "ron", "slo": "slk", "bh": "bih"}


features = {}
# Add if adding target-side features for MT
#features["eng"] = {}
#features["eng"]["word_vocab"] = en_v
#features["eng"]["subword_vocab"] = en_sub_v


with open('../transfer_corpus/langs.txt') as f:
	langs = f.read().splitlines()

dict_langs = {'am': 'amh', 'ar': 'ara', 'da': 'dan', 'de': 'deu', 'en': 'eng', 'es': 'spa', 'fa': 'fas',
				  'fi': 'fin', 'fr': 'fra', 'hu': 'hun', 'id': 'ind', 'it': 'ita', 'ko': 'kor', 'lv': 'lav',
				  'ne': 'nep', 'nl': 'nld', 'no': 'nor', 'pt': 'por', 'ro': 'ron', 'sv': 'swe', 'uk': 'ukr',
				  'zh': 'zho'}

langs = [lang if len(lang)> 2 else dict_langs[lang] for lang in langs]

for filename in os.listdir(dataset_dir):

	if 'DS_Store' in filename: continue
	# print(filename)
	temp = filename.split(".")
	language = temp[-1]

	if language not in langs: continue

	# Get number of lines in training data
	# filename = "ted-train.orig."+temp[0]
	#if len(language) == 2:
	#	language = LETTER_CODES[language]
	filename = os.path.join(dataset_dir, filename)
	bashCommand = "wc -l " + filename
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	lines = int(output.strip().split()[0])
	print(language, filename + " " + str(lines))

	all_words = get_vocab(filename)

	c = Counter(all_words)
	key = filename.split('-')[0] + '_'+language
	features[key] = {}
	features[key]["lang"] = language
	features[key]["dataset_size"] = lines

	unique = list(c)
	# Get number of types and tokens
	#features[key]["token_number"] = len(all_words)
	#features[key]["type_number"] = len(unique)
	features[key]["word_vocab"] = unique
	#features[key]["type_token_ratio"] = features[key]["type_number"] / float(features[key]["token_number"])


indexed = "indexed/NER"
outputfile = os.path.join(indexed, "news_ner.npy")
np.save(outputfile, features)

