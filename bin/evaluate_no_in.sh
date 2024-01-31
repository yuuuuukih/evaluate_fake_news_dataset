cd ~/evaluate_fake_news_dataset/src

SUB_DIR="diff7_rep3"

MODES=("pre_target_timeline" "all_timeline")
PRED_BYS=("cls" "target")
SPLIT_RATIOS=(25 50 75 100)

for MODE in "${MODES[@]}"
do
    for PRED_BY in "${PRED_BYS[@]}"
    do
        for SPLIT_RATIO in "${SPLIT_RATIOS[@]}"
        do
            echo -e "\n Running evaluate.py $MODE / $PRED_BY / $SPLIT_RATIO / no_in"
            poetry run python evaluate.py \
                --sub_dir $SUB_DIR \
                --mode $MODE \
                --pred_by $PRED_BY \
                --split_ratio $SPLIT_RATIO \
                --no_in
        done
    done
done

cd ~/evaluate_fake_news_dataset
