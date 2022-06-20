CUDA_VISIBLE_DEVICES=7 python interact.py \
    --config_name strat \
    --inputter_name strat \
    --seed 3 \
    --load_checkpoint /home/zhengchujie/EmotionalSupportConversation/DATA/strat.strat/2021-07-31145609.3e-05.16.1gpu/epoch-2.pt \
    --fp16 false \
    --max_src_len 150 \
    --max_tgt_len 50 \
    --max_length 50 \
    --min_length 10 \
    --temperature 0.7 \
    --top_k 0 \
    --top_p 0.9 \
    --num_beams 1 \
    --repetition_penalty 1 \
    --no_repeat_ngram_size 3
