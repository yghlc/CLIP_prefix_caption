#!/usr/bin/env python
# Filename: utils.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 02 July, 2025
"""

import os, sys

import clip
import torch

sys.path.insert(0, os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS'))
import basic_src.io_function as io_function

def save_image_captions(image_list, caption_list, save_path):
    image_caption_dict = {image: caption for image, caption in zip(image_list, caption_list)}
    io_function.save_dict_to_txt_json(save_path, image_caption_dict)
    print(f'Saved captions to {save_path}')



def get_output_name(model_path,clip_mode_path=None,clip_model_type=None, image_list_txt=None):
    model_name = io_function.get_name_no_ext(model_path)

    output_name = model_name

    if clip_mode_path is not None:
        clip_mode_name = io_function.get_name_no_ext(clip_mode_path)
        output_name += '-' + clip_mode_name

    if clip_model_type is not None:
        clip_model_type = clip_model_type.replace('/','-')
        output_name += '-' + clip_model_type
    if image_list_txt is not None:
        img_list_name = io_function.get_name_no_ext(image_list_txt)
        output_name += '-' + img_list_name

    return output_name+'.json'


def get_image_path_list(input, file_extension):
    image_path_list = []
    if input.endswith(".txt"):
        # read file name from the txt file
        image_path_list = io_function.read_list_from_txt(input)
    elif os.path.isfile(input):
        image_path_list.append(image_path_list)
    elif os.path.isdir(input):
        # read files name from a folder
        image_path_list = io_function.get_file_list_by_pattern(image_path_list, f'*{file_extension}')
    else:
        raise IOError(f'Cannot recognize the input: {image_path_list}')

    return image_path_list

def load_clip_model(device,model_type='ViT-B/32',trained_model=None, b_with_state_dict=False):
    model, preprocess = clip.load(model_type, device=device)
    # load trained model
    if trained_model is not None:
        if os.path.isfile(trained_model):
            checkpoint = torch.load(open(trained_model, 'rb'), map_location="cpu")
            if b_with_state_dict:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint,strict=True)

    return model, preprocess

def parser_options(parser):
    parser.add_option("-s", "--trained_model",
                      action="store", dest="trained_model",
                      help="the trained model")

    parser.add_option("-c", "--trained_clip_model",
                      action="store", dest="trained_clip_model",
                      help="the trained CLIP model")

    parser.add_option("-t", "--clip_model_type",
                      action="store", dest="clip_model_type",
                      help="the CLIP model type ")

    parser.add_option("-o", "--output_file",
                      action="store", dest="output_file",
                      help="the file to save the generated captions, if not set, will save to image_captions.json")

    parser.add_option("-e", "--image_ext", action="store", dest="ext",
                      help="the extension of the image files, like .tif or .jpg (don't miss the dot), \
                      need this when the input is a folder")

def main():
    pass


if __name__ == '__main__':
    main()
