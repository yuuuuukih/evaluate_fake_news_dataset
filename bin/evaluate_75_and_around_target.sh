cd ~/evaluate_fake_news_dataset/src

SUB_DIR=diff7_rep3

# ==============
# re-run 75%
# ==============
MODES=("base" "pre_target_timeline" "all_timeline")

for MODE in "${MODES[@]}"
do
    echo -e "\n Running evaluate.py $MODE / cls / 75 / with_in"
    CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python evaluate.py \
        --sub_dir $SUB_DIR \
        --mode $MODE \
        --pred_by cls \
        --split_ratio 75

    echo -e "\n Running evaluate.py $MODE / cls / 75 / no_in"
    CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python evaluate.py \
        --sub_dir $SUB_DIR \
        --mode $MODE \
        --pred_by cls \
        --split_ratio 75 \
        --no_in
done


# ==============
# around_target
# ==============

PRED_BYS=("cls" "target")
SPLIT_RATIOS=(25 50 75 100)

for PRED_BY in "${PRED_BYS[@]}"
do
    for SPLIT_RATIO in "${SPLIT_RATIOS[@]}"
    do
        echo -e "\n Running evaluate.py around_target / $PRED_BY / $SPLIT_RATIO / with_in"
        CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python evaluate.py \
            --sub_dir $SUB_DIR \
            --mode around_target \
            --pred_by $PRED_BY \
            --split_ratio $SPLIT_RATIO

        echo -e "\n Running evaluate.py around_target / $PRED_BY / $SPLIT_RATIO / no_in"
        CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run python evaluate.py \
            --sub_dir $SUB_DIR \
            --mode around_target \
            --pred_by $PRED_BY \
            --split_ratio $SPLIT_RATIO \
            --no_in
    done
done


cd ~/evaluate_fake_news_dataset
