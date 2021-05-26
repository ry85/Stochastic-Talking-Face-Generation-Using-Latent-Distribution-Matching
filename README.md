# Stochastic-Talking-Face-Generation
This is the official [PyTorch] implementation of our Interspeech'20 paper "Stochastic Talking Face Generation Using Latent Distribution Matching".

Paper Link:
https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1823.pdf

Demo:
https://user-images.githubusercontent.com/26645173/119601305-837caf80-be06-11eb-83e5-b19ed9216ac9.mp4

It shows two different predicted video for same reference image and audio input.

# Training step
CUDA_VISIBLE_DEVICES=0 python3 train.py --batch_size 5 --lr 0.0002 --beta 0.000001 --z_dim 24 --model vgg --image_width 128 --data_threads 4

To restore training:
CUDA_VISIBLE_DEVICES=0 python3 train.py --model_dir <path_to_model.pth>


