CUDA_VISIBLE_DEVICES=1 python3 train_bert_seed.py --model_name senticgcn_bert --dataset rest14 --save True --learning_rate 2e-5 --seed 39 --batch_size 16 --hidden_dim 768
