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

def main(load_model=False):
    batch_size = 64       # batch size
    vocab_threshold = 6        # minimum word count threshold
    vocab_from_file = True    # if True, load existing vocab file
    embed_size = 512           # dimensionality of image and word embeddings
    hidden_size = 512          # number of features in hidden state of the RNN decoder
    num_epochs = 5             # number of training epochs
    save_every = 5             # determines frequency of saving model weights
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
                             vocab_from_file=vocab_from_file,
                             cocoapi_loc='/nobackup/users/vinhle/nlp/opt/')

    # Set the total number of training steps per epoch.
    total_step = int(math.ceil(60000 / data_loader.batch_sampler.batch_size))
    # total_step = int(math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size) / 3)
    # The size of the vocabulary.
    vocab_size = len(data_loader.dataset.vocab)

    # Initialize the encoder and decoder.
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    model_data = None
    if load_model:
        model_data = torch.load(load_model)
        encoder.load_state_dict(model_data['encoder_state_dict'])
        decoder.load_state_dict(model_data['decoder_state_dict'])

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
    if load_model:
        optimizer.load_state_dict(model_data['optimizer_state_dict'])
    # Open the training log file.

    f = open(log_file, 'w')
    for epoch in tqdm(range(1, num_epochs + 1),position=0, leave=True):

        for i_step in tqdm(range(1, total_step + 1),position=1, leave=True):

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
        # Get training statistics.
        stats = 'Epoch [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
            epoch, num_epochs, loss.item(), np.exp(loss.item()))

        # Print training statistics to file.
        f.write(stats + '\n')
        f.flush()
        # Print training statistics to file.


        # Save the weights.
        if epoch % save_every == 0:
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'model_data-{epoch}.tar')

    # Close the training log file.
    f.close()


if __name__ == '__main__':
    benchmark = True
    # load_models = ['models/encoder-20.pkl','models/decoder-20.pkl']
    load_models = 'model_data-25.tar'
    main(load_models)