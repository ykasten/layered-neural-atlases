from implicit_neural_networks import IMLP
from unwrap_utils import load_input_data
import time
import torch
import numpy as np
import sys

import argparse
from evaluate import evaluate_model
import os

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path




def main(training_folder, frames_folder, mask_rcnn_folder,  output_folder, video_name,runinng_command):
    # read config:
    with open("%s/config.json" % training_folder) as f:
        config = json.load(f)

    maximum_number_of_frames = config["maximum_number_of_frames"]
    resx = np.int64(config["resx"])
    resy = np.int64(config["resy"])

    positional_encoding_num_alpha = config["positional_encoding_num_alpha"]

    number_of_channels_atlas = config["number_of_channels_atlas"]
    number_of_layers_atlas = config["number_of_layers_atlas"]

    number_of_channels_alpha = config["number_of_channels_alpha"]
    number_of_layers_alpha = config["number_of_layers_alpha"]
    uv_mapping_scale = config["uv_mapping_scale"]
    use_positional_encoding_mapping1 = config["use_positional_encoding_mapping1"]
    number_of_positional_encoding_mapping1 = config["number_of_positional_encoding_mapping1"]
    number_of_layers_mapping1 = config["number_of_layers_mapping1"]
    number_of_channels_mapping1 = config["number_of_channels_mapping1"]

    use_positional_encoding_mapping2 = config["use_positional_encoding_mapping2"]
    number_of_positional_encoding_mapping2 = config["number_of_positional_encoding_mapping2"]
    number_of_layers_mapping2 = config["number_of_layers_mapping2"]
    number_of_channels_mapping2 = config["number_of_channels_mapping2"]
    derivative_amount = config["derivative_amount"]
    data_folder = Path(config["data_folder"])
    vid_name = data_folder.name
    vid_root = data_folder.parent
    optical_flows_mask, video_frames, _, mask_frames, _, _, _, optical_flows = load_input_data(
            resy, resx, maximum_number_of_frames, data_folder, True, True, vid_root, vid_name)
    number_of_frames = video_frames.shape[3]

    # load networks' weights:
    model_F_mapping1 = IMLP(
        input_dim=3,
        output_dim=2,
        hidden_dim=number_of_channels_mapping1,
        use_positional=use_positional_encoding_mapping1,
        positional_dim=number_of_positional_encoding_mapping1,
        num_layers=number_of_layers_mapping1,
        skip_layers=[]).to(device)
    model_F_mapping2 = IMLP(
        input_dim=3,
        output_dim=2,
        hidden_dim=number_of_channels_mapping2,
        use_positional=use_positional_encoding_mapping2,
        positional_dim=number_of_positional_encoding_mapping2,
        num_layers=number_of_layers_mapping2,
        skip_layers=[]).to(device)

    model_F_atlas = IMLP(
        input_dim=2,
        output_dim=3,
        hidden_dim=number_of_channels_atlas,
        use_positional=True,
        positional_dim=10,
        num_layers=number_of_layers_atlas,
        skip_layers=[4, 7]).to(device)

    model_alpha = IMLP(
        input_dim=3,
        output_dim=1,
        hidden_dim=number_of_channels_alpha,
        use_positional=True,
        positional_dim=positional_encoding_num_alpha,
        num_layers=number_of_layers_alpha,
        skip_layers=[]).to(device)

    checkpoint = torch.load("%s/checkpoint" % training_folder)

    model_F_atlas.load_state_dict(checkpoint["F_atlas_state_dict"])
    model_F_atlas.eval()
    model_F_atlas.to(device)

    model_F_mapping1.load_state_dict(checkpoint["model_F_mapping1_state_dict"])
    model_F_mapping1.eval()
    model_F_mapping1.to(device)

    model_F_mapping2.load_state_dict(checkpoint["model_F_mapping2_state_dict"])
    model_F_mapping2.eval()
    model_F_mapping2.to(device)

    model_alpha.load_state_dict(checkpoint["model_F_alpha_state_dict"])
    model_alpha.eval()
    model_alpha.to(device)

    folder_time = time.time()


    Path(os.path.join(output_folder, '%s_%06d' % (video_name, folder_time))).mkdir(parents=True, exist_ok=True)
    file1 = open("%s/runinng_command" % os.path.join(output_folder, '%s_%06d' % (video_name, folder_time)), "w")
    file1.write(runinng_command)
    file1.close()
    start_iteration = checkpoint["iteration"]
    # run evaluation:
    evaluate_model(model_F_atlas,resx,resy,number_of_frames,model_F_mapping1,model_F_mapping2,model_alpha,video_frames,os.path.join(output_folder, '%s_%06d' % (video_name, folder_time)),start_iteration,mask_frames,0,0,video_name,derivative_amount,uv_mapping_scale,optical_flows,optical_flows_mask,device,save_checkpoint=False,show_atlas_alpha=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--trained_model_folder', type=str,
                        help='the folder that contains the trained model')


    parser.add_argument('--data_folder', type=str,
                        help='the folder that contains the masks produced by Mask-RCNN and the images ')
    parser.add_argument('--video_name', type=str,
                        help='the name of the video that should be evaluated')

    parser.add_argument('--output_folder', type=str,
                        help='the folder that will contains the output evaluation ')


    args = parser.parse_args()

    training_folder = args.trained_model_folder
    video_name = args.video_name

    frames_folder = os.path.join(args.data_folder, video_name)
    mask_rcnn_folder = os.path.join(args.data_folder, video_name) + "_maskrcnn"
    output_folder = args.output_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)


    main(training_folder, frames_folder, mask_rcnn_folder, output_folder, video_name, ' '.join(sys.argv))
