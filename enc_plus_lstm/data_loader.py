import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json
from torchvision import transforms


def get_loader(transform,
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='opt/'):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    
    assert mode in ['train', 'test', 'eval'], "mode must be one of 'train' or 'test'."
    if vocab_from_file==False: assert mode=='train', "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if mode == 'train':
        if vocab_from_file==True: assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = cocoapi_loc + 'cocoapi/train2014/'
        annotations_file = cocoapi_loc + 'cocoapi/annotations/captions_train2014.json'

    if mode == 'test' or 'eval':
        assert batch_size==1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/val2017/')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/annotations/captions_val2017.json')

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    if mode == 'train':
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader

class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        if self.mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]

        elif self.mode == 'eval':
            eval_info = json.loads(open(annotations_file).read())
            self.img_id_to_captions = {}
            self.img_id_to_img = {}
            for img_dict in eval_info['annotations']:
                img_id = img_dict['image_id']
                self.img_id_to_captions.setdefault(img_id,[])
                self.img_id_to_captions[img_id].append(img_dict['caption'])
            for img_dict in eval_info['images']:
                self.img_id_to_img[img_dict['id']] = img_dict['file_name']
            self.ids = list(self.img_id_to_img.keys())
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]
        
    def __getitem__(self, index):
        # obtain image and caption if in training mode
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']

            # Convert image to tensor and pre-process using transform
            image = Image.open(self.img_folder + path).convert('RGB')
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # return pre-processed image and caption tensors
            return image, caption

        # obtain image if in test mode
        elif self.mode == 'test':
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # return original image and pre-processed image tensor
            return orig_image, image

        elif self.mode == 'eval':
            ann_id = self.ids[index]
            captions = self.img_id_to_captions[ann_id]
            img_p = self.img_id_to_img[ann_id]

            # Convert image to tensor and pre-process using transform
            image = Image.open(self.img_folder + img_p).convert('RGB')
            image = self.transform(image)

            # return pre-processed image and caption tensors
            return image, captions


    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == 'train' or self.mode == 'eval':
            return len(self.ids)
        else:
            return len(self.paths)


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),  # smaller edge of image resized to 256
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))])
    batch_size = 1
    vocab_threshold = 6  # minimum word count threshold
    vocab_from_file = True
    x = get_loader(transform=transform_train,
                   mode='eval',
                   batch_size=batch_size,
                   vocab_threshold=vocab_threshold,
                   vocab_from_file=vocab_from_file,
                   cocoapi_loc='C:/Users/vinhl/Documents/School/68610/img_captioning/opt/')
    images, captions = next(iter(x))

    # annotations_file = 'C:/Users/vinhl/Documents/School/68610/img_captioning/opt/cocoapi/combined_dataset/captions.json'
    # data = json.loads(open(annotations_file).read())
    # print(data['images'][0:5])
    # print(data['annotations'][0:5])
