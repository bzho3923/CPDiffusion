
CUDA_VISIBLE_DEVICES=0 python protein_DIFF/inference.py \
--ckpt result/xx.pt \
--target_protein xx.pt \
--target_protein_dir xx/process/ \
--fix_pos_file dataset/xx.fix.txt \
--gen_num 100 \
--output_dir result/predict