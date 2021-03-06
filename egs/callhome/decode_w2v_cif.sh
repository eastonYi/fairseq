. ../path.sh

gpu=$1
label_type=char
beam=1
DATA_DIR=data/ma/hkust_style_char
data_name=test
MODEL_PATH=exp/finetune_w2v_lm_teacher_forcing_2.1/checkpoint_best.pt
RESULT_DIR=exp/finetune_w2v_lm_teacher_forcing_2.1/decode_w2v_cif_beam${beam}_${data_name}

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu \
python $MAIN_ROOT/examples/speech_recognition/infer.py $DATA_DIR \
    --gen-subset $data_name \
    --path $MODEL_PATH \
    --labels $label_type \
    --results-path $RESULT_DIR \
    --task audio_cif --not-add-ctc-blank --w2l-decoder cif_lm_decoder \
    --criterion qua_ce_acc --iscn \
    --beam ${beam} --remove-bpe $label_type --max-tokens 600000
