#!/bin/bash
# set -e

if [[ ! -e InterpretEval ]]; then
  git clone git@github.com:neulab/InterpretEval.git
fi

full_names[0]="Naija_pidgin"
full_names[1]="amharic"
full_names[2]="igbo"
full_names[3]="luganda"
full_names[4]="wolof"
full_names[5]="hausa"
full_names[6]="kinyarwanda"
full_names[7]="luo"
full_names[8]="swahili"
full_names[9]="yoruba"

lang_codes[0]="pcm"
lang_codes[1]="amh"
lang_codes[2]="ibo"
lang_codes[3]="lug"
lang_codes[4]="wol"
lang_codes[5]="hau"
lang_codes[6]="kin"
lang_codes[7]="luo"
lang_codes[8]="swa"
lang_codes[9]="yor"

for i in $(seq 0 9); do
  mycode=${lang_codes[$i]}
  myname=${full_names[$i]}
  echo "$myname $mycode"
  masadir=../
  iedir=InterpretEval/interpretEval/data/ner/masakhane-$mycode
  mkdir -p $iedir/{data,results}
  cp $masadir/data/$mycode/{train,dev,test}.txt $iedir/data
  if [[ -e $masadir/entity_analysis/XLM-R/${mycode}_xlmr_test_predictions.txt ]]; then
    python add_gold.py $masadir/entity_analysis/XLM-R/${mycode}_xlmr_test_predictions.txt $masadir/data/$mycode/test.txt > $iedir/results/${mycode}_xlmr_test.txt
  else
    echo "WARNING: $masadir/entity_analysis/XLM-R/${mycode}_xlmr_test_predictions.txt doesn't exist"
  fi
  if [[ -e $masadir/entity_analysis/mBERT/${mycode}_bert_test_predictions.txt ]]; then
    python add_gold.py $masadir/entity_analysis/mBERT/${mycode}_bert_test_predictions.txt $masadir/data/$mycode/test.txt > $iedir/results/${mycode}_mbert_test.txt
  else
    echo "WARNING: $masadir/entity_analysis/mBERT/${mycode}_bert_test_predictions.txt doesn't exist"
  fi
  if [[ -e $masadir/entity_analysis/biLSTM_CRF/test.${mycode}_model ]]; then
    cp $masadir/entity_analysis/biLSTM_CRF/test.${mycode}_model $iedir/results/${mycode}_bilstmcrf_test.txt
  else
    echo "WARNING: $masadir/entity_analysis/biLSTM_CRF/test.${mycode}_model doesn't exist"
  fi
done

cd InterpretEval/interpretEval

for i in $(seq 0 9); do
  mycode=${lang_codes[$i]}
  bash ../../run_task_ner_masakhane.sh $mycode bilstmcrf xlmr
  bash ../../run_task_ner_masakhane.sh $mycode mbert xlmr
done
