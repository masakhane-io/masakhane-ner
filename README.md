## **MasakhaNER: Named Entity Recognition for African Languages**

This repository contains the code for [training NER models](https://github.com/masakhane-io/masakhane-ner/tree/main/code) for the two MasakhaNER projects:

* **[MasakhaNER 1.0](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00416/107614/MasakhaNER-Named-Entity-Recognition-for-African)**: NER dataset for 10 African languages (Amharic, Hausa, Igbo, Kinyarwanda, Luganda, Luo, Nigerian-Pidgin, Swahili, Wolof and Yorùbá). This annotation for the dataset was performed by volunteers from the [Masakhane](https://www.masakhane.io/) community, leveraging the [participatory research design](https://aclanthology.org/2020.findings-emnlp.195/) that has been shown to be successful for building machine translation models . 

* **[MasakhaNER 2.0](https://aclanthology.org/2022.emnlp-main.298/)**: An expansion of MasakhaNER 1.0 to 20 African languages, the dataset includes all MasakhaNER 1.0, except for Amharic, and 11 new languages from West Africa (Bambara, Ewe, Fon, and Twi), Central Africa (Ghomala) and Southern Africa (Chichewa, Setwana, chiShona, isiXhosa, and isiZulu). The project has been generously funded by [Lacuna Fund](https://lacunafund.org/announcing-new-datasets-for-african-languages-2020-natural-language-processing-nlp-awardees/). More details about the project can be found [here](https://github.com/masakhane-io/lacuna_pos_ner). 


### Required dependencies
* python
  * [transformers](https://pypi.org/project/transformers/) : state-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.
  * [seqeval](https://pypi.org/project/seqeval/) : testing framework for sequence labeling.
  * [ptvsd](https://pypi.org/project/ptvsd/) : remote debugging server for Python support in Visual Studio and Visual Studio Code.

```bash
pip install transformers seqeval ptvsd
```

### License information
The code is based on HuggingFace implementation (License: Apache 2.0).

The license of the NER dataset is in [CC-BY-4.0-NC](https://creativecommons.org/licenses/by-nc/4.0/), the monolingual data have difference licenses depending on the news website license. 


### Dataset information
* MasakhaNER 1.0 can be found in CoNLL format in [data/](https://github.com/masakhane-io/masakhane-ner/tree/main/data) or [huggingface datasets](https://huggingface.co/datasets/masakhaner)
* MasakhaNER 2.0 can be found in CoNLL format in [MasakhaNER2.0/data/](https://github.com/masakhane-io/masakhane-ner/tree/main/MasakhaNER2.0/data) or [huggingface datasets](https://huggingface.co/datasets/masakhane/masakhaner2)
* MasakhaNER-X is an aggregation of MasakhaNER 1.0 and MasakhaNER 2.0 datasets for 20 African languages. The dataset is not in CoNLL format. The input is the original raw text while the output is byte-level span annotations. The dataset is in [xtreme-up/MasakhaNER-X](https://github.com/masakhane-io/masakhane-ner/tree/main/xtreme-up/MasakhaNER-X/). For more details, see the ([XTREME-UP](https://github.com/google-research/xtreme-up)).


### Load dataset on HuggingFace
```
from datasets import load_dataset
data = load_dataset('masakhaner', 'yor')
data = load_dataset('masakhane/masakhaner2', 'yor')
```

### African NER model
We provide a single multilingual NER model for all the 20 African languages on [Huggingface Model Hub](https://huggingface.co/masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0)
```
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
model = AutoModelForTokenClassification.from_pretrained("masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Emir of Kano turban Zhang wey don spend 18 years for Nigeria"
ner_results = nlp(example)
print(ner_results)

```

### Predict the best transfer language for zero-shot adaptation
If your language is not supported by our model, you can predict the best transfer language to adapt from that would give the best performance. This also support non-African languages because we trained the [ranking model](https://github.com/masakhane-io/masakhane-ner/blob/main/MasakhaNER2.0/ranking_languages/pretrained/NER/lgbm_model_all.txt) on both African and non-African languages (in Europe and Asia). More details can be found [MasakhaNER2.0/](https://github.com/masakhane-io/masakhane-ner/tree/main/MasakhaNER2.0) directory and in the [paper](https://aclanthology.org/2022.emnlp-main.298/). 

To run the code, follow the instructions on [LangRank](https://github.com/neulab/langrank) based on this [paper](https://aclanthology.org/P19-1301/), and install the requirements. Run code in [ranking_languages/](https://github.com/masakhane-io/masakhane-ner/tree/main/MasakhaNER2.0/ranking_languages). 

This is an example for Sesotho. 
```
export LANG=sot
python3 langrank_predict.py -o ranking_data/datasets/ner-train.orig.$LANG -s ranking_data/datasets_spm/ner-train.orig.spm.$LANG -l $LANG -n 3 -t NER -m best

#1. ranking_data/datasets/ner_tsn : score=1.96
#	1. Entity overlap : score=1.55; 
#	2. GEOGRAPHIC : score=0.99; 
#	3. INVENTORY : score=0.66
#2. ranking_data/datasets/ner_swa : score=-0.19
#	1. INVENTORY : score=0.70; 
#	2. Transfer over target size ratio : score=0.51; 
#	3. GEOGRAPHIC : score=0.49
#3. ranking_data/datasets/ner_nya : score=-0.57
#	1. INVENTORY : score=0.85; 
#	2. GEOGRAPHIC : score=0.64; 
#	3. GENETIC : score=0.34
```


### BibTeX entry and citation info
If you make use of the MasakhaNER 1.0 dataset, please cite the our TACL paper. For the MasakhaNER 2.0, please cite our EMNLP paper :

```
@article{10.1162/tacl_a_00416,
    author = {Adelani, David Ifeoluwa and Abbott, Jade and Neubig, Graham and D’souza, Daniel and Kreutzer, Julia and Lignos, Constantine and Palen-Michel, Chester and Buzaaba, Happy and Rijhwani, Shruti and Ruder, Sebastian and Mayhew, Stephen and Azime, Israel Abebe and Muhammad, Shamsuddeen H. and Emezue, Chris Chinenye and Nakatumba-Nabende, Joyce and Ogayo, Perez and Anuoluwapo, Aremu and Gitau, Catherine and Mbaye, Derguene and Alabi, Jesujoba and Yimam, Seid Muhie and Gwadabe, Tajuddeen Rabiu and Ezeani, Ignatius and Niyongabo, Rubungo Andre and Mukiibi, Jonathan and Otiende, Verrah and Orife, Iroro and David, Davis and Ngom, Samba and Adewumi, Tosin and Rayson, Paul and Adeyemi, Mofetoluwa and Muriuki, Gerald and Anebi, Emmanuel and Chukwuneke, Chiamaka and Odu, Nkiruka and Wairagala, Eric Peter and Oyerinde, Samuel and Siro, Clemencia and Bateesa, Tobius Saul and Oloyede, Temilola and Wambui, Yvonne and Akinode, Victor and Nabagereka, Deborah and Katusiime, Maurice and Awokoya, Ayodele and MBOUP, Mouhamadane and Gebreyohannes, Dibora and Tilaye, Henok and Nwaike, Kelechi and Wolde, Degaga and Faye, Abdoulaye and Sibanda, Blessing and Ahia, Orevaoghene and Dossou, Bonaventure F. P. and Ogueji, Kelechi and DIOP, Thierno Ibrahima and Diallo, Abdoulaye and Akinfaderin, Adewale and Marengereke, Tendai and Osei, Salomey},
    title = "{MasakhaNER: Named Entity Recognition for African Languages}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {9},
    pages = {1116-1131},
    year = {2021},
    month = {10},
    abstract = "{We take a step towards addressing the under- representation of the African continent in NLP research by bringing together different stakeholders to create the first large, publicly available, high-quality dataset for named entity recognition (NER) in ten African languages. We detail the characteristics of these languages to help researchers and practitioners better understand the challenges they pose for NER tasks. We analyze our datasets and conduct an extensive empirical evaluation of state- of-the-art methods across both supervised and transfer learning settings. Finally, we release the data, code, and models to inspire future research on African NLP.1}",
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00416},
    url = {https://doi.org/10.1162/tacl\_a\_00416},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00416/1966201/tacl\_a\_00416.pdf},
}


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
