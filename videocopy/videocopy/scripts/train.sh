DATASET=VCSL
MODEL=model_super

python train.py \
       --model-file transvcl/weights/${MODEL}.pth \
       --feat-dir data/${DATASET}/features/ \
       --test-file data/${DATASET}/pair_file.csv \
       --save-file results/${MODEL}/${DATASET}/result.json