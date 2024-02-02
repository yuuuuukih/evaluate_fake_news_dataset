cd ~/evaluate_fake_news_dataset/src

SUB_DIR=diff7_rep3
PRED_BY=cls

CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python run.py \
    --sub_dir $SUB_DIR \
    --mode base \
    --pred_by $PRED_BY \
    --split_ratio 75 \

CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python run.py \
    --sub_dir $SUB_DIR \
    --mode base \
    --pred_by $PRED_BY \
    --split_ratio 75 \
    --no_in

CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python run.py \
    --sub_dir $SUB_DIR \
    --mode pre_target_timeline \
    --pred_by $PRED_BY \
    --split_ratio 75 \

CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python run.py \
    --sub_dir $SUB_DIR \
    --mode pre_target_timeline \
    --pred_by $PRED_BY \
    --split_ratio 75 \
    --no_in

CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python run.py \
    --sub_dir $SUB_DIR \
    --mode all_timeline \
    --pred_by $PRED_BY \
    --split_ratio 75 \

CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python run.py \
    --sub_dir $SUB_DIR \
    --mode all_timeline \
    --pred_by $PRED_BY \
    --split_ratio 75 \
    --no_in


cd ~/evaluate_fake_news_dataset
