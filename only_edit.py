from implicit_neural_networks import IMLP
import matplotlib.image as mpimg
import time
from scipy.interpolate import griddata
import torch
import numpy as np
import sys
import imageio
import cv2
from PIL import Image
import argparse

from evaluate import get_high_res_texture, get_colors,get_mapping_area
import os

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path


def apply_edit(model_F_atlas, resx, resy, number_of_frames, model_F_mapping1, model_F_mapping2, model_alpha,
               video_frames,
               output_folder_final, mask_frames, vid_name,
               evaluate_all_pixels=False,texture_edit_im1=None, texture_edit_im2=None,
               alpha_im1=None, alpha_im2=None):

    larger_dim = np.maximum(resx, resy)

    # get relevant working crops from the atlases for atlas discretization
    minx = 0
    miny = 0
    edge_size = 1
    maxx2, minx2, maxy2, miny2, edge_size2 = get_mapping_area(model_F_mapping2, model_alpha, mask_frames > -1, larger_dim,
                                                              number_of_frames,
                                                              torch.tensor([-0.5, -0.5]),device, invert_alpha=True)

    edited_tex1, texture_orig1 = get_high_res_texture(
        1000,
        0,1, 0, 1, model_F_atlas,device)

    texture_orig1_alpha = torch.zeros((1000, 1000, 4))
    texture_orig1_alpha[:, :, :3] = texture_orig1
    edited_tex2, texture_orig2 = get_high_res_texture(
        1000,
        minx2, minx2 + edge_size2, miny2, miny2 + edge_size2,  model_F_atlas,device
               )

    edited_tex1_only_edit = torch.from_numpy(texture_edit_im1)
    # save the given edits:
    imageio.imwrite(
        "%s/texture_edit_user1.png" % output_folder_final,(
        np.concatenate((texture_edit_im1,alpha_im1[:,:,np.newaxis]),axis=2)*255).astype(np.uint8))
    edited_tex1 = torch.from_numpy(1 - alpha_im1).unsqueeze(-1) * texture_orig1 + torch.from_numpy(
        alpha_im1).unsqueeze(-1) * texture_edit_im1

    edited_tex1_only_edit = torch.cat((edited_tex1_only_edit, torch.from_numpy(alpha_im1).unsqueeze(-1)), dim=-1)
    edited_tex2_only_edit = torch.from_numpy(texture_edit_im2)
    imageio.imwrite(
        "%s/texture_edit_user2.png" % output_folder_final,(
        np.concatenate((texture_edit_im2,alpha_im2[:,:,np.newaxis]),axis=2)*255).astype(np.uint8))
    edited_tex2 = torch.from_numpy(1 - alpha_im2).unsqueeze(-1) * texture_orig2 + torch.from_numpy(
        alpha_im2).unsqueeze(-1) * texture_edit_im2

    edited_tex2_only_edit = torch.cat((edited_tex2_only_edit, torch.from_numpy(alpha_im2).unsqueeze(-1)), dim=-1)

    alpha_reconstruction = np.zeros((resy, resx, number_of_frames))

    masks1 = np.zeros((edited_tex1.shape[0], edited_tex1.shape[1]))
    masks2 = np.zeros((edited_tex2.shape[0], edited_tex2.shape[1]))

    only_mapped_texture = np.zeros((resy, resx, 4, number_of_frames))
    only_mapped_texture2 = np.zeros((resy, resx, 4, number_of_frames))
    rgb_edit_video = np.zeros((resy, resx, 3, number_of_frames))

    with torch.no_grad():
        for f in range(number_of_frames):
            print(f)
            if evaluate_all_pixels:
                relis_i, reljs_i = torch.where(torch.ones(resy, resx) > 0)
            else:
                relis_i, reljs_i = torch.where(torch.ones(resy, resx) > 0)

            relisa = np.array_split(relis_i.numpy(), np.ceil(relis_i.shape[0] / 100000))
            reljsa = np.array_split(reljs_i.numpy(), np.ceil(relis_i.shape[0] / 100000))


            for i in range(len(relisa)):
                relis = torch.from_numpy(relisa[i]).unsqueeze(1) / (larger_dim / 2) - 1
                reljs = torch.from_numpy(reljsa[i]).unsqueeze(1) / (larger_dim / 2) - 1

                uv_temp1 = model_F_mapping1(
                    torch.cat((reljs, relis,
                               (f / (number_of_frames / 2.0) - 1) * torch.ones_like(relis)),
                              dim=1).to(device))
                uv_temp2 = model_F_mapping2(
                    torch.cat((reljs, relis,
                               (f / (number_of_frames / 2.0) - 1) * torch.ones_like(relis)),
                              dim=1).to(device))

                alpha = 0.5 * (model_alpha(torch.cat((reljs, relis,
                                                      (f / (number_of_frames / 2.0) - 1) * torch.ones_like(relis)),
                                                     dim=1).to(device)) + 1.0)
                alpha = alpha * 0.99
                alpha = alpha + 0.001

                uv_temp1 = uv_temp1.detach().cpu()
                uv_temp2 = uv_temp2.detach().cpu()
                # sample the edit colors from the edited atlases in the relevant uv coordinates
                rgb_only_edit, pointsx1, pointsy1, relevant1_only_edit = get_colors(1000, minx, minx + edge_size, miny,
                                                                      miny + edge_size,
                                                                      uv_temp1[:, 0] * 0.5 + 0.5,
                                                                      uv_temp1[:, 1] * 0.5 + 0.5,
                                                                      edited_tex1_only_edit)

                rgb_only_edit2, pointsx2, pointsy2, relevant2_only_edit = get_colors(1000,
                                                                       minx2, minx2 + edge_size2, miny2,
                                                                       miny2 + edge_size2,
                                                                       uv_temp2[:, 0] * 0.5 - 0.5,
                                                                       uv_temp2[:, 1] * 0.5 - 0.5,
                                                                       edited_tex2_only_edit)

                try:
                    masks2[np.ceil(pointsy2).astype((np.int64)), np.ceil(pointsx2).astype((np.int64))] = 1
                    masks2[np.floor(pointsy2).astype((np.int64)), np.floor(pointsx2).astype((np.int64))] = 1
                    masks2[np.floor(pointsy2).astype((np.int64)), np.ceil(pointsx2).astype((np.int64))] = 1
                    masks2[np.ceil(pointsy2).astype((np.int64)), np.floor(pointsx2).astype((np.int64))] = 1
                except Exception:
                    pass

                try:
                    masks1[np.ceil(pointsy1).astype((np.int64)), np.ceil(pointsx1).astype((np.int64))] = 1
                    masks1[np.floor(pointsy1).astype((np.int64)), np.floor(pointsx1).astype((np.int64))] = 1
                    masks1[np.floor(pointsy1).astype((np.int64)), np.ceil(pointsx1).astype((np.int64))] = 1
                    masks1[np.ceil(pointsy1).astype((np.int64)), np.floor(pointsx1).astype((np.int64))] = 1
                except Exception:
                    pass
                alpha_reconstruction[relisa[i], reljsa[i], f] = alpha[:, 0].detach().cpu(
                ).numpy()

                # save the video pixels of the edits from the two atlases
                only_mapped_texture[relisa[i][relevant1_only_edit], reljsa[i][relevant1_only_edit], :,
                f] = rgb_only_edit

                only_mapped_texture2[relisa[i][relevant2_only_edit], reljsa[i][relevant2_only_edit], :,
                f] = rgb_only_edit2
            # see details in Section 3.4 in the paper
            foreground_edit_cur = only_mapped_texture[:, :, :3, f] # denoted in the paper by c_{ef}
            foreground_edit_cur_alpha = only_mapped_texture[:, :, 3, f][:, :, np.newaxis] # denoted by \alpha_{ef}

            background_edit_cur = only_mapped_texture2[:, :, :3, f] # denoted in the paper by c_{eb}
            background_edit_cur_alpha = only_mapped_texture2[:, :, 3, f][:, :, np.newaxis] # denoted in the paper by \alpha_{eb}

            video_frame_cur = video_frames[:, :, :, f].cpu().clone().numpy()  # denoted in the paper by \bar{c}_{b}

            # Equation (15):
            video_frame_cur_edited1 = foreground_edit_cur * (foreground_edit_cur_alpha) + video_frame_cur * (
                        1 - foreground_edit_cur_alpha) #\bar{c}_b
            video_frame_cur_edited2 = background_edit_cur * (background_edit_cur_alpha) + video_frame_cur * (
                    1 - background_edit_cur_alpha) #\bar{c}_f

            cur_alpha = alpha_reconstruction[:, :, f][:, :, np.newaxis]

            # Equation (16):
            foreground_edit_output = video_frame_cur_edited1 * cur_alpha + (1 - cur_alpha) * video_frame_cur_edited2

            rgb_edit_video[:, :, :, f] = foreground_edit_output


    mpimg.imsave("%s/texture_edit1.png" % output_folder_final,
                 (masks1[:, :, np.newaxis] * edited_tex1.numpy() * (255)).astype(np.uint8))
    mpimg.imsave("%s/texture_orig1.png" % output_folder_final,
                 (masks1[:, :, np.newaxis] *texture_orig1.numpy() * (255)).astype(np.uint8))


    mpimg.imsave("%s/texture_edit2.png" % output_folder_final,
                 (masks2[:, :, np.newaxis] * edited_tex2.numpy() * (255)).astype(np.uint8))

    mpimg.imsave("%s/texture_orig2.png" % output_folder_final,
                 (masks2[:, :, np.newaxis]*texture_orig2.numpy() * (255)).astype(np.uint8))

    writer_edit = imageio.get_writer(
        "%s/edited_%s.mp4" % (output_folder_final, vid_name),
        fps=10)

    # Save the edit video
    for i in range(number_of_frames):
        print(i)
        writer_edit.append_data((rgb_edit_video[:, :, :, i] * (255)).astype(np.uint8))

    writer_edit.close()



def texture_edit_from_frame_edit(edit_frame, f, model_F_mapping1, model_F_mapping2, model_alpha, number_of_frames,
                                 mask_frames, edit_frame_foreground, edit_frame_background,device):
    resx = edit_frame.shape[1]
    resy = edit_frame.shape[0]
    larger_dim = np.maximum(resx, resy)        

    minx = 0
    miny = 0
    edge_size = 1
    maxx2, minx2, maxy2, miny2, edge_size2 = get_mapping_area(model_F_mapping2, model_alpha, mask_frames > -1, larger_dim,
                                                              number_of_frames,
                                                              torch.tensor([-0.5, -0.5]),device, invert_alpha=True)

    relis_i, reljs_i = torch.where(torch.ones(edit_frame.shape[0], edit_frame.shape[1]) > 0)

    relisa = np.array_split(relis_i.numpy(), np.ceil(relis_i.shape[0] / 100000))
    reljsa = np.array_split(reljs_i.numpy(), np.ceil(relis_i.shape[0] / 100000))

    inds1 = []
    colors1 = []
    inds2 = []
    colors2 = []

    for i in range(len(relisa)):
        relis = torch.from_numpy(relisa[i]).unsqueeze(1) / (larger_dim / 2) - 1
        reljs = torch.from_numpy(reljsa[i]).unsqueeze(1) / (larger_dim / 2) - 1
        # map frame edit to texture coordinates using the mapping networks
        uv_temp1 = model_F_mapping1(
            torch.cat((reljs, relis,
                       (f / (number_of_frames / 2.0) - 1) * torch.ones_like(relis)),
                      dim=1).to(device))
        uv_temp2 = model_F_mapping2(
            torch.cat((reljs, relis,
                       (f / (number_of_frames / 2.0) - 1) * torch.ones_like(relis)),
                      dim=1).to(device))

        finalcoords1 = (((uv_temp1 * 0.5 + 0.5) - torch.tensor([[minx, miny]]).to(device)) / edge_size) * 1000

        finalcoords2 = (((uv_temp2 * 0.5 - 0.5) - torch.tensor([[minx2, miny2]]).to(device)) / edge_size2) * 1000

        alpha = 0.5 * (model_alpha(torch.cat((reljs, relis,
                                              (f / (number_of_frames / 2.0) - 1) * torch.ones_like(relis)),
                                             dim=1).to(device)) + 1.0)
        alpha = alpha * 0.99
        alpha = alpha + 0.001

        inds1.append(finalcoords1.detach().cpu().numpy())
        inds2.append(finalcoords2.detach().cpu().numpy())
        # the alpha values tell us how to split the RGBA values from the frames to the two atlas edits:
        colors1.append(edit_frame[relisa[i], reljsa[i], :] * alpha.detach().cpu().numpy())

        colors2.append(edit_frame[relisa[i], reljsa[i], :] * (1 - alpha).detach().cpu().numpy())

    # We have target (non integer) coordinates (inds1,inds2) and target color and we use them to
    # render 2 1000x1000 RGBA atlas edits
    inds1 = np.concatenate(inds1)
    inds2 = np.concatenate(inds2)

    colors1 = np.concatenate(colors1)
    colors2 = np.concatenate(colors2)
    xv, yv = np.meshgrid(np.linspace(0, 999, 1000), np.linspace(0, 999, 1000))

    edit_im1 = griddata(inds1, colors1, (xv, yv), method='linear')
    edit_im1[np.isnan(edit_im1)] = 0

    edit_im2 = griddata(inds2, colors2, (xv, yv), method='linear')
    edit_im2[np.isnan(edit_im2)] = 0

    if edit_frame_background:
        edit_im1[:, :, 3] = 0 # do not apply any edit on the foreground
    elif edit_frame_foreground:
        edit_im2[:, :, 3] = 0 # do not apply any edit on the background
    return edit_im1, edit_im2


def main(training_folder, frame_edit, frames_folder, mask_rcnn_folder, frame_edit_file, edit_tex1_file, edit_tex2_file,
         frame_edit_index, output_folder, video_name, edit_frame_foreground, edit_frame_background,runinng_command):
    # Read the config of the trained model
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

    use_positional_encoding_mapping1 = config["use_positional_encoding_mapping1"]
    number_of_positional_encoding_mapping1 = config["number_of_positional_encoding_mapping1"]
    number_of_layers_mapping1 = config["number_of_layers_mapping1"]
    number_of_channels_mapping1 = config["number_of_channels_mapping1"]

    use_positional_encoding_mapping2 = config["use_positional_encoding_mapping2"]
    number_of_positional_encoding_mapping2 = config["number_of_positional_encoding_mapping2"]
    number_of_layers_mapping2 = config["number_of_layers_mapping2"]
    number_of_channels_mapping2 = config["number_of_channels_mapping2"]


    data_folder = Path(frames_folder)
    maskrcnn_dir = Path(mask_rcnn_folder)
    input_files = sorted(list(data_folder.glob('*.jpg')) + list(data_folder.glob('*.png')))
    mask_files = sorted(list(maskrcnn_dir.glob('*.jpg')) + list(maskrcnn_dir.glob('*.png')))

    number_of_frames = np.minimum(maximum_number_of_frames,len(input_files))
    # read video frames and maskRCNN masks
    video_frames = torch.zeros((resy, resx, 3, number_of_frames))
    mask_frames = torch.zeros((resy, resx, number_of_frames))

    for i in range(number_of_frames):
        file1 = input_files[i]
        im = np.array(Image.open(str(file1))).astype(np.float64) / 255.
        mask = np.array(Image.open(str(mask_files[i]))).astype(np.float64) / 255.
        mask = cv2.resize(mask, (resx, resy), cv2.INTER_NEAREST)
        mask_frames[:, :, i] = torch.from_numpy(mask)
        video_frames[:, :, :, i] = torch.from_numpy(cv2.resize(im[:, :, :3], (resx, resy)))


    # Define MLPs
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

    if frame_edit:
        edit_frame = imageio.imread(frame_edit_file)[:, :, :] / 255.0
        frame_number = frame_edit_index
        # get texture edits from frame edit
        edit_im1, edit_im2 = texture_edit_from_frame_edit(edit_frame, frame_number, model_F_mapping1, model_F_mapping2,
                                                          model_alpha, number_of_frames, mask_frames,
                                                          edit_frame_foreground, edit_frame_background,device)

        alpha_im1 = edit_im1[:, :, 3]
        edit_im1 = edit_im1[:, :, :3]
        alpha_im2 = edit_im2[:, :, 3]
        edit_im2 = edit_im2[:, :, :3]

        edited_frame = video_frames[:, :, :, frame_number].numpy()
        edited_frame = edit_frame[:, :, :3] * (edit_frame[:, :, 3][:, :, np.newaxis]) + edited_frame[:, :, :3] * (
                    1 - edit_frame[:, :, 3][:, :, np.newaxis])
    else:
        edit_im1 = imageio.imread(edit_tex1_file)[:, :, :3] / 255.0
        alpha_im1 = imageio.imread(edit_tex1_file)[:, :, 3] / 255.0

        edit_im2 = imageio.imread(edit_tex2_file)[:, :, :3] / 255.0
        alpha_im2 = imageio.imread(edit_tex2_file)[:, :, 3] / 255.0
    output_folder_final = output_folder_final = os.path.join(output_folder, '%s_%06d' % (video_name, folder_time))

    Path(output_folder_final).mkdir(parents=True, exist_ok=True)

    file1 = open("%s/runinng_command" %output_folder_final, "w")
    file1.write(runinng_command)
    file1.close()
    apply_edit(model_F_atlas, resx, resy, number_of_frames, model_F_mapping1, model_F_mapping2, model_alpha,
               video_frames, output_folder_final, mask_frames, video_name,
               texture_edit_im1=edit_im1,
               texture_edit_im2=edit_im2, alpha_im1=alpha_im1, alpha_im2=alpha_im2)
    if frame_edit:
        imageio.imwrite("%s/the_edited_frame.png" % output_folder_final, (edited_frame*255).astype(np.uint8))
        imageio.imwrite("%s/the_input_edit_frame.png" % output_folder_final, (edit_frame*255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--trained_model_folder', type=str,
                        help='the folder that contains the trained model')

    parser.add_argument('--data_folder', type=str,
                        help='the folder that contains the masks produced by Mask-RCNN and the images ')
    parser.add_argument('--video_name', type=str,
                        help='the name of the video that should be edited')

    parser.add_argument('--output_folder', type=str,
                        help='the folder that will contains the output editing ')

    parser.add_argument('--use_edit_frame', type=bool, nargs='?',
                        const=True,
                        help='if true, the code expects an edit of one frame', default=False)

    parser.add_argument('--edit_frame_foreground', type=bool, nargs='?',
                        const=True,
                        help='if true, the edit is applied only on the foreground', default=False)

    parser.add_argument('--edit_frame_background', type=bool, nargs='?',
                        const=True,
                        help='if true, the edit is applied only on the background', default=False)

    parser.add_argument('--edit_frame_index', type=int,
                        help='if use_edit_frame==true, the code needs the frame index that should be edited')

    parser.add_argument('--edit_frame_path', type=str,
                        help='if use_edit_frame==true, the code needs the edit for the frame')

    parser.add_argument('--edit_foreground_path', type=str,
                        help='the path to the foreground texture edit')
    parser.add_argument('--edit_background_path', type=str,
                        help='the path to the background texture edit')

    args = parser.parse_args()

    training_folder = args.trained_model_folder
    video_name = args.video_name

    frames_folder = os.path.join(args.data_folder, video_name)
    mask_rcnn_folder = os.path.join(args.data_folder, video_name) + "_maskrcnn"
    output_folder = args.output_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    frame_edit = args.use_edit_frame
    edit_frame_foreground = args.edit_frame_foreground
    edit_frame_background = args.edit_frame_background
    if frame_edit:
        frame_edit_file = args.edit_frame_path
        frame_edit_index = args.edit_frame_index
        edit_tex1_file = 0
        edit_tex2_file = 0
    else:
        frame_edit_file = 0
        frame_edit_index = 0
        edit_tex1_file = args.edit_foreground_path
        edit_tex2_file = args.edit_background_path

    main(training_folder, frame_edit, frames_folder, mask_rcnn_folder, frame_edit_file, edit_tex1_file, edit_tex2_file,
         frame_edit_index, output_folder, video_name, edit_frame_foreground, edit_frame_background,' '.join(sys.argv))
