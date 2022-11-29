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

def main():
    batch_size = 64       # batch size
    vocab_threshold = 6        # minimum word count threshold
    vocab_from_file = True    # if True, load existing vocab file
    embed_size = 512           # dimensionality of image and word embeddings
    hidden_size = 512          # number of features in hidden state of the RNN decoder
    num_epochs = 3             # number of training epochs
    save_every = 1             # determines frequency of saving model weights
    log_file = 'training_log.txt'       # name of file with saved training loss and perplexity

    transform_train = transforms.Compose([
        transforms.Resize((256,256)),                          # smaller edge of image resized to 256
        transforms.RandomCrop(224),                      # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
        transforms.ToTensor(),                           # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])


    data_loader = get_loader(transform=transform_train,
                             mode='train',
                             batch_size=batch_size,
                             vocab_threshold=vocab_threshold,
                             vocab_from_file=vocab_from_file)

    # The size of the vocabulary.
    vocab_size = len(data_loader.dataset.vocab)

    # Initialize the encoder and decoder.
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    # Move models to GPU if CUDA is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device being used', device)
    encoder.to(device)
    decoder.to(device)

    # Define the loss function.
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    # Specify the learnable parameters of the model.
    params = list(decoder.parameters()) + list(encoder.embed.parameters())

    # Define the optimizer.
    optimizer = torch.optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), eps=1e-8)

    # Set the total number of training steps per epoch.
    total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

    # Open the training log file.
    f = open(log_file, 'w')


    for epoch in tqdm(range(1, num_epochs + 1)):

        for i_step in tqdm(range(1, total_step + 1),position=0, leave=True):

            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler

            # Obtain the batch.
            images, captions = next(iter(data_loader))

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions = captions.to(device)

            # Zero the gradients.
            decoder.zero_grad()
            encoder.zero_grad()

            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions)

            # Calculate the batch loss.
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

        # Get training statistics.
        stats = 'Epoch [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
        epoch, num_epochs, loss.item(), np.exp(loss.item()))

        # Print training statistics to file.
        f.write(stats + '\n')
        f.flush()



        # Save the weights.
        if epoch % save_every == 0:
            torch.save(decoder.state_dict(), f'models/decoder-{epoch}.pkl')
            torch.save(encoder.state_dict(), f'models/encoder-{epoch}.pkl')

    # Close the training log file.
    f.close()


if __name__ == '__main__':
    benchmark = True
    main()