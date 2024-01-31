cd ~/evaluate_fake_news_dataset/src

SUB_DIR="diff7_rep3"
MODE="base"
PRED_BY="cls"

echo $SUB_DIR
echo $MODE
echo $PRED_BY

echo -e "\n Running evaluate.py base / cls / 25 / no_in"
poetry run python evaluate.py \
    --sub_dir $SUB_DIR \
    --mode $MODE \
    --pred_by $PRED_BY \
    --split_ratio 25 \
    --no_in

echo -e "\n Running evaluate.py base / cls / 50 / no_in"
poetry run python evaluate.py \
    --sub_dir $SUB_DIR \
    --mode $MODE \
    --pred_by $PRED_BY \
    --split_ratio 50 \
    --no_in

echo -e "\n Running evaluate.py base / cls / 75 / no_in"
poetry run python evaluate.py \
    --sub_dir $SUB_DIR \
    --mode $MODE \
    --pred_by $PRED_BY \
    --split_ratio 75 \
    --no_in

echo -e "\n Running evaluate.py base / cls / 100 / no_in"
poetry run python evaluate.py \
    --sub_dir $SUB_DIR \
    --mode $MODE \
    --pred_by $PRED_BY \
    --split_ratio 100 \
    --no_in

cd ~/evaluate_fake_news_dataset
