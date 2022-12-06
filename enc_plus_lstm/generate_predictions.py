import sys
from pycocotools.coco import COCO
from data_loader import get_loader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import os
from model import EncoderCNN, DecoderRNN
import argparse


def main(img_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    data_loader = get_loader(transform=transform,
                             mode='train',
                             batch_size=64,
                             vocab_threshold=6,
                             vocab_from_file=True,
                             cocoapi_loc='/nobackup/users/vinhle/nlp/opt/')
    def clean_sentence(output):
        sentence = ""
        for i in output:
            word = data_loader.dataset.vocab.idx2word[i]
            if (word == data_loader.dataset.vocab.start_word):
                continue
            elif (word == data_loader.dataset.vocab.end_word):
                break
            else:
                sentence = sentence + " " + word
        return sentence

    # Select appropriate values for the Python variables below.
    embed_size = 512
    hidden_size = 512

    # The size of the vocabulary.
    vocab_size = len(data_loader.dataset.vocab)
    # Initialize the encoder and decoder, and set each to inference mode.
    encoder = EncoderCNN(embed_size)
    encoder.eval()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    decoder.eval()

    # Load the trained weights.
    model_data = torch.load(model_path)
    encoder.load_state_dict(model_data['encoder_state_dict'])
    decoder.load_state_dict(model_data['decoder_state_dict'])

    # Move models to GPU if CUDA is available.
    encoder.to(device)
    decoder.to(device)


    def generate(im_p):
        img = Image.open(im_p).convert('RGB')
        img = transform(img)
        img_shape = list(img.shape)
        img = img.view([1] + img_shape)
        img = img.to(device)
        features = encoder(img).unsqueeze(1)
        output = decoder.sample(features)
        sentence = clean_sentence(output)
        return sentence

    print(generate(img_path))

    img_path = input("Img path")
    cont_flag = True
    while cont_flag:
        print(generate(img_path))
        img_path = input("Img path: ")
        if img_path == "stop":
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path')
    parser.add_argument('model_path')
    args = parser.parse_args()
    print(main(args.img_path, args.model_path))
