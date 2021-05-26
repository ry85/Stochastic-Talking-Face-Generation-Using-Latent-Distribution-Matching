# Stochastic-Talking-Face-Generation
This is the official [PyTorch] implementation of our Interspeech'20 paper "Stochastic Talking Face Generation Using Latent Distribution Matching".

# Training step
CUDA_VISIBLE_DEVICES=0 python3 train.py --batch_size 5 --lr 0.0002 --beta 0.000001 --z_dim 24 --model vgg --image_width 128 --data_threads 4

To restore training:
CUDA_VISIBLE_DEVICES=0 python3 train.py --model_dir <path_to_model.pth>


