
CUDA_VISIBLE_DEVICES=1 python protein_DIFF/inference.py \
--ckpt result/weight/Jun_5_ago_dataset=CATH_result_lr=0.0002_wd=0.0_dp=0.08_hidden=256_noisy_type=uniform_embed_ss=False_89474.pt \
--target_protein dataset/Ago/AGO_050_model_3_ptm.pt \
--target_protein_dir dataset/Ago/process/ \
--gen_num 100 \
--output_dir result/predict