RESULT_DIR=$1
NAME=$2
v=best

/home/easton/files/sctk-2.4.10/bin/sclite \
-r ${RESULT_DIR}/ref.word-checkpoint_${v}.pt-${NAME}.txt trn \
-h ${RESULT_DIR}/hypo.word-checkpoint_${v}.pt-${NAME}.txt \
-i rm -c NOASCII -s -o all stdout > ${RESULT_DIR}/${NAME}.result1.wrd.txt

vi ${RESULT_DIR}/${NAME}.result1.wrd.txt
