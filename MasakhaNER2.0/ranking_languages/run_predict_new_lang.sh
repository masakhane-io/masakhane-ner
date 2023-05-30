
for TGT_LANG in ful lin orm sot ven run tir
do 
	export LANG=$TGT_LANG
	python3 langrank_predict.py -o ranking_data/datasets/ner-train.orig.$LANG -s ranking_data/datasets_spm/ner-train.orig.spm.$LANG -l $LANG -n 3 -t NER -m best >> results_new_lang/$LANG.txt

done
