### Evaluate Open-MAGVIT2 Pretrain 262144 NPU
python evaluation_original_reso.py --config_file configs/Open-MAGVIT2/npu/pretrain_lfqgan_256_262144.yaml --ckpt_path ../upload_ckpts/Open-MAGVIT2/pretrain_256_262144/pretrain256_262144.ckpt --original_reso --model Open-MAGVIT2 --batch_size 1

### Evaluate Open-MAGVIT2 Pretrain 16384 NPU
# python evaluation_original_reso.py --config_file configs/Open-MAGVIT2/npu/pretrain_lfqgan_256_16384.yaml --ckpt_path ../upload_ckpts/Open-MAGVIT2/pretrain_256_16384/pretrain256_16384.ckpt --model Open-MAGVIT2 --original_reso --batch_size 1

### Evaluate Open-MAGVIT2 Pretrain 262144 GPU
# python evaluation_original_reso.py --config_file configs/Open-MAGVIT2/gpu/pretrain_lfqgan_256_262144.yaml --ckpt_path ../upload_ckpts/Open-MAGVIT2/pretrain_256_262144/pretrain256_262144.ckpt --original_reso --model Open-MAGVIT2 --batch_size 1

### Evaluate Open-MAGVIT2 Pretrain 16384 GPU
# python evaluation_original_reso.py --config_file configs/Open-MAGVIT2/gpu/pretrain_lfqgan_256_16384.yaml --ckpt_path ../upload_ckpts/Open-MAGVIT2/pretrain_256_16384/pretrain256_16384.ckpt --original_reso --model Open-MAGVIT2 --batch_size 1