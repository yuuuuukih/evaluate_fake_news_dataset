cd ~/evaluate_fake_news_dataset/src

SUB_DIR=diff7_rep3
MODE=base
PRED_BY=cls

CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python run.py \
    --sub_dir $SUB_DIR \
    --mode $MODE \
    --pred_by $PRED_BY \
    --split_ratio 25 \
    --no_in

CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python run.py \
    --sub_dir $SUB_DIR \
    --mode $MODE \
    --pred_by $PRED_BY \
    --split_ratio 50 \
    --no_in

CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python run.py \
    --sub_dir $SUB_DIR \
    --mode $MODE \
    --pred_by $PRED_BY \
    --split_ratio 75 \
    --no_in

CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python run.py \
    --sub_dir $SUB_DIR \
    --mode $MODE \
    --pred_by $PRED_BY \
    --split_ratio 100 \
    --no_in

cd ~/evaluate_fake_news_dataset
