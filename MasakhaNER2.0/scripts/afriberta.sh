LANG=nya
for j in 1 2 3 4 5
do
	export MAX_LENGTH=164
	export BERT_MODEL=castorini/afriberta_large
	export OUTPUT_DIR=${LANG}_afriberta
	export TEXT_RESULT=test_result$j.txt
	export TEXT_PREDICTION=test_predictions$j.txt
	export BATCH_SIZE=32
	export NUM_EPOCHS=20
	export SAVE_STEPS=10000
	export SEED=$j

	CUDA_VISIBLE_DEVICES=3 python3 ../../code/train_ner.py --data_dir ../data/${LANG}/ \
	--model_type xlmroberta \
	--model_name_or_path $BERT_MODEL \
	--output_dir $OUTPUT_DIR \
	--test_result_file $TEXT_RESULT \
	--test_prediction_file $TEXT_PREDICTION \
	--max_seq_length  $MAX_LENGTH \
	--num_train_epochs $NUM_EPOCHS \
	--per_gpu_train_batch_size $BATCH_SIZE \
	--save_steps $SAVE_STEPS \
	--seed $SEED \
	--do_train \
	--do_eval \
	--do_predict \
	--overwrite_output_dir
done
