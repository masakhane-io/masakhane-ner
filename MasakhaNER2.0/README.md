## [MasakhaNER: Named Entity Recognition for African Languages](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00416/107614/MasakhaNER-Named-Entity-Recognition-for-African)

This repository contains the code for [training NER models](https://github.com/masakhane-io/masakhane-ner/tree/main/code), scripts to [analyze the NER model predictions](https://github.com/masakhane-io/masakhane-ner/tree/main/analysis_scripts) and the [NER datasets](https://github.com/masakhane-io/masakhane-ner/tree/main/data) for all the 10 languages listed below. 

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

### Volunteers
----------------
| Language | Volunteer names |
|----------|-----------------|
| Amharic | Seid Muhie Yimam, Musie Meressa, Israel Abebe, Degaga Wolde, Henok Tilaye, Dibora Haile  |
| Hausa  | Shamsudden Muhammad, Tajuddeen Rabiu Gwadabe, Emmanuel Anebi, Idris Abdulmumin|
| Igbo  | Ignatius Ezeani, Chris Emezue, Chukwuneke Chiamaka, Nkiru Odu, Amaka, Isaac |
| Kinyarwanda | Rubungo Andre Niyongabo, Happy Buzaaba |
|Luganda   |  Joyce Nabende, Jonathan Mukiibi, Eric Peter Kigaye, Ivan Ssenkungu, Ibrahim Mbabaali, Batista Tobius, Maurice Katusiime, Deborah Nabagereka, Tobius Saolo |
| Luo   | Perez Ogayo, Verrah Otiende |
| Naija Pidgin | Orevaoghene Ahia, Kelechi Ogueji, Adewale	Akinfaderin, Aremu Adeola Jr., Iroro Orife, Temi Oloyede, Samuel Abiodun Oyerinde, Victor Akinode   |
| Swahili | Catherine Gitau, Verrah Otiende, Davis David, Clemencia Siro, Yvonne Wambui, Gerald Muriuki  |
| Wolof | [Abdoulaye Diallo](https://github.com/abdoulsn), [Thierno Ibrahim Diop](https://github.com/bayethiernodiop), and [Derguene Mbaye](https://github.com/DerXter), Samba Ngom, Mouhamadane Mboup  |
| Yorùbá | David Adelani, Mofetoluwa Adeyemi, Jesujoba Alabi, Tosin Adewumi, Ayodele Awokoya |

If you make use of this dataset, please cite us:

### BibTeX entry and citation info
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
```
