gpu=$1
DATA_BIN=/data3/easton/data/TEXT/data-bin/wikitext-103
# MODEL=exp/transformer_wikitext-103/checkpoint_best.pt
MODEL=/home/easton/projects/fairseq/egs/language_model/exp/transformer_wikitext-103/adaptive_lm_wiki103.v2/model.pt

 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu  fairseq-eval-lm $DATA_BIN \
    --path $MODEL \
    --max-sentences 1 \
    --tokens-per-sample 512 \
    --context-window 400

# | Evaluated 245569 tokens in 56.1s (4379.02 tokens/s)
# | Loss: 3.4164, Perplexity: 30.46
