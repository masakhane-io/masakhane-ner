## [MasakhaNER 2.0: Africa-centric Transfer Learning for Named Entity Recognition](https://aclanthology.org/2022.emnlp-main.298/)

This repository contains the code for [training NER models](https://github.com/masakhane-io/masakhane-ner/tree/main/code), [sample scripts for the experiments]([https://github.com/masakhane-io/masakhane-ner/tree/main/analysis_scripts](https://github.com/masakhane-io/masakhane-ner/tree/main/MasakhaNER2.0/scripts)) and the [NER datasets]([https://github.com/masakhane-io/masakhane-ner/tree/main/data](https://github.com/masakhane-io/masakhane-ner/tree/main/MasakhaNER2.0/data)) for all the 20 languages listed below. 

The code is based on HuggingFace implementation (License: Apache 2.0).

The license of the NER dataset is in [CC-BY-4.0-NC](https://creativecommons.org/licenses/by-nc/4.0/), the monolingual data have difference licenses depending on the news website license. 

### Required dependencies
* python
  * [transformers](https://pypi.org/project/transformers/) : state-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.
  * [seqeval](https://pypi.org/project/seqeval/) : testing framework for sequence labeling.
  * [ptvsd](https://pypi.org/project/ptvsd/) : remote debugging server for Python support in Visual Studio and Visual Studio Code.

```bash
pip install transformers seqeval ptvsd
```

### Languages
----------------
For all languages, there are three splits.

The original splits were named `train`, `dev` and `test` and they correspond to the `train`, `validation` and `test` splits.

The splits have the following sizes :

| Language        | train | validation | test  |
|-----------------|------:|-----------:|------:|
| Bambara         |  4463 |        638 |  1274 |
| Ghomala         |  3384 |        483 |   966 |
| Ewe             |  3505 |        501 |  1001 |
| Fon.            |  4343 |        621 |  1240 |
| Hausa           |  5716 |        816 |  1633 |
| Igbo            |  7634 |       1090 |  2181 |
| Kinyarwanda     |  7825 |       1118 |  2235 |
| Luganda         |  4942 |        706 |  1412 |
| Luo             |  5161 |        737 |  1474 |
| Mossi           |  4532 |        648 |  1613 |
| Nigerian-Pidgin |  5646 |        806 |  1294 |
| Chichewa        |  6250 |        893 |  1785 |
| chiShona        |  6207 |        887 |  1773 |
| Kiswahili       |  6593 |        942 |  1883 |
| Setswana        |  3289 |        499 |   996 |
| Akan/Twi        |  4240 |        605 |  1211 |
| Wolof           |  4593 |        656 |  1312 |
| isiXhosa        |  5718 |        817 |  1633 |
| Yoruba          |  6877 |        983 |  1964 |
| isiZulu         |  5848 |        836 |  1670 |




If you make use of this dataset, please cite us:

### BibTeX entry and citation info
```
@inproceedings{adelani-etal-2022-masakhaner,
    title = "{M}asakha{NER} 2.0: {A}frica-centric Transfer Learning for Named Entity Recognition",
    author = "Adelani, David  and
      Neubig, Graham  and
      Ruder, Sebastian  and
      Rijhwani, Shruti  and
      Beukman, Michael  and
      Palen-Michel, Chester  and
      Lignos, Constantine  and
      Alabi, Jesujoba  and
      Muhammad, Shamsuddeen  and
      Nabende, Peter  and
      Dione, Cheikh M. Bamba  and
      Bukula, Andiswa  and
      Mabuya, Rooweither  and
      Dossou, Bonaventure F. P.  and
      Sibanda, Blessing  and
      Buzaaba, Happy  and
      Mukiibi, Jonathan  and
      Kalipe, Godson  and
      Mbaye, Derguene  and
      Taylor, Amelia  and
      Kabore, Fatoumata  and
      Emezue, Chris Chinenye  and
      Aremu, Anuoluwapo  and
      Ogayo, Perez  and
      Gitau, Catherine  and
      Munkoh-Buabeng, Edwin  and
      Memdjokam Koagne, Victoire  and
      Tapo, Allahsera Auguste  and
      Macucwa, Tebogo  and
      Marivate, Vukosi  and
      Elvis, Mboning Tchiaze  and
      Gwadabe, Tajuddeen  and
      Adewumi, Tosin  and
      Ahia, Orevaoghene  and
      Nakatumba-Nabende, Joyce  and
      Mokono, Neo Lerato  and
      Ezeani, Ignatius  and
      Chukwuneke, Chiamaka  and
      Oluwaseun Adeyemi, Mofetoluwa  and
      Hacheme, Gilles Quentin  and
      Abdulmumin, Idris  and
      Ogundepo, Odunayo  and
      Yousuf, Oreen  and
      Moteu, Tatiana  and
      Klakow, Dietrich",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.298",
    pages = "4488--4508",
    abstract = "African languages are spoken by over a billion people, but they are under-represented in NLP research and development. Multiple challenges exist, including the limited availability of annotated training and evaluation datasets as well as the lack of understanding of which settings, languages, and recently proposed methods like cross-lingual transfer will be effective. In this paper, we aim to move towards solutions for these challenges, focusing on the task of named entity recognition (NER). We present the creation of the largest to-date human-annotated NER dataset for 20 African languages. We study the behaviour of state-of-the-art cross-lingual transfer methods in an Africa-centric setting, empirically demonstrating that the choice of source transfer language significantly affects performance. While much previous work defaults to using English as the source language, our results show that choosing the best transfer language improves zero-shot F1 scores by an average of 14{\%} over 20 languages as compared to using English.",
}
```
