import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from pycocotools.coco import COCO
from data_loader import get_loader
from torch.backends.cudnn import benchmark
from model import EncoderCNN, DecoderRNN
import math
import torch.utils.data as data
import numpy as np
import os
from tqdm import tqdm
import sys
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from cider import Cider
import nltk



def main(model_path=False):
    batch_size = 1       # batch size
    vocab_threshold = 6        # minimum word count threshold
    vocab_from_file = True    # if True, load existing vocab file
    embed_size = 512           # dimensionality of image and word embeddings
    hidden_size = 512          # number of features in hidden state of the RNN decoder
    num_epochs = 1000            # number of training epochs

    transform = transforms.Compose([
        transforms.Resize((256,256)),                          # smaller edge of image resized to 256
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])


    data_loader = get_loader(transform=transform,
                             mode='eval',
                             batch_size=batch_size,
                             vocab_threshold=vocab_threshold,
                             vocab_from_file=vocab_from_file,
                             cocoapi_loc='C:/Users/vinhl/Documents/School/68610/img_captioning/opt/')

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


    # The size of the vocabulary.
    vocab_size = len(data_loader.dataset.vocab)

    # Initialize the encoder and decoder.
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device being used', device)
    encoder.to(device)
    decoder.to(device)
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    def caption_to_tensor(caption):
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(data_loader.dataset.vocab(data_loader.dataset.vocab.start_word))
        caption.extend([data_loader.dataset.vocab(token) for token in tokens])
        caption.append(data_loader.dataset.vocab(data_loader.dataset.vocab.end_word))
        return torch.Tensor([caption]).long()

    def generate(img, cap):
        perp = []
        img = img.to(device)
        features = encoder(img)
        for c in cap:
            c = caption_to_tensor(c).to(device)
            otherput = decoder(features, c)
            perp.append(np.exp(criterion(otherput.view(-1, vocab_size), c.view(-1)).item()))
        features = features.unsqueeze(1)
        output = decoder.sample(features)
        sentence = clean_sentence(output)
        return sentence, np.min(perp)

    total_bleu = []
    total_cider = []
    perplexity = []
    for iteration in tqdm(range(num_epochs),position=0, leave=True):
        # Obtain the batch.
        images, captions = next(iter(data_loader))
        candidate, perp = generate(images, captions)
        captions = [i[0] for i in captions]
        captions_bleu = [i.split() for i in captions]
        candidate_bleu = candidate.split()
        bleu_score = sentence_bleu(captions_bleu, candidate_bleu, weights=(.95, .05, 0, 0), smoothing_function = SmoothingFunction().method3)
        hypo_for_image = {1: [candidate]}
        ref_for_image = {1: captions}
        c = Cider()
        cider_score, _ = c.compute_score(ref_for_image, hypo_for_image)
        total_bleu.append(bleu_score)
        total_cider.append(cider_score)
        perplexity.append(perp)
    return total_bleu, total_cider, perplexity



if __name__ == '__main__':
    bleu , cider, perplexity = main('mscoco_model.tar')
    print('bleuscore:', np.mean(bleu))
    print('ciderscore:', np.mean(cider))
    print('perplexity:', np.mean(perplexity))
    '''
    bleuscore: 0.4302091876986542
ciderscore: 0.6868682982377469'''



