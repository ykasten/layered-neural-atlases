# Layered Neural Atlases for Consistent Video Editing
### [Project Page](https://layered-neural-atlases.github.io/) | [Paper](https://arxiv.org/pdf/2109.11418.pdf) 

<p align="center">
  <img width="100%" src="media/teaser_lucia.gif"/>
</p>

This repository contains an implementation for the SIGGRAPH Asia 2021 paper <a href="https://arxiv.org/pdf/2109.11418.pdf">Layered Neural Atlases for Consistent Video Editing</a>.

The paper introduces the first approach for neural video unwrapping using an end-to-end optimized interpretable and semantic atlas-based representation, which facilitates easy and intuitive editing in the atlas domain.

## Installation Requirements
The code is compatible with Python 3.7 and PyTorch 1.6. 

You can create an anaconda environment called `neural_atlases` with the required dependencies by running:
```
conda create --name neural_atlases python=3.7 
conda activate neural_atlases 
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy  scikit-image tqdm  opencv -c pytorch
pip install imageio-ffmpeg gdown
python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
```

## Data convention
The code expects 3 folders for each video input, e.g. for a video of 50 frames named "blackswan":
1. `data/blackswan`: A folder of video frames containing image files in the following convention: `blackswan/00000.jpg`,`blackswan/00001.jpg`,...,`blackswan/00049.jpg`  (as in the  [DAVIS](https://davischallenge.org/)  dataset).
2. `data/blackswan_flow`: A folder with forward and backward optical flow files in the following convention: `blackswan_flow/00000.jpg_00001.jpg.npy`,`blackswan_flow/00001.jpg_00000.jpg`,...,`blackswan_flow/00049.jpg_00048.jpg.npy`.
3. `data/blackswan_maskrcnn`: A folder with rough masks (created by [Mask-RCNN](https://arxiv.org/abs/1703.06870) or any other way) containing files in the following convention: `blackswan_maskrcnn/00000.jpg`,`blackswan_maskrcnn/00001.jpg`,...,`blackswan_maskrcnn/00049.jpg`

For a few examples of DAVIS sequences run:
```
gdown https://drive.google.com/uc?id=1WipZR9LaANTNJh764ukznXXAANJ5TChe
unzip data.zip
```

#### Masks extraction
Given only the video frames folder `data/blackswan` it is possible to extract the [Mask-RCNN](https://arxiv.org/abs/1703.06870) masks (and create the required folder `data/blackswan_maskrcnn`) by running: 
```
python preprocess_mask_rcnn.py --vid-path data/blackswan --class_name bird
```
where --class_name determines the COCO class name of the sought foreground object. It is also possible to choose the first instance retrieved by Mask-RCNN by using `--class_name anything`. This is usefull for cases where Mask-RCNN gets correct masks with wrong classes as in the "libby" video:
```
python preprocess_mask_rcnn.py --vid-path data/libby --class_name anything
```

#### Optical flows extraction
Furthermore, the optical flow folder can be extracted using [RAFT](https://arxiv.org/abs/2003.12039). 
For linking RAFT into the current project run:
```
git submodule update --init
cd thirdparty/RAFT/
./download_models.sh
cd ../..
```

For extracting the optical flows (and creating the required folder `data/blackswan_flow`) run: 
```
python preprocess_optical_flow.py --vid-path data/blackswan --max_long_edge 768
```

## Pretrained models
For downloading a sample set of our pretrained models together with sample edits run:
```
gdown https://drive.google.com/uc?id=10voSCdMGM5HTIYfT0bPW029W9y6Xij4D
unzip pretrained_models.zip
```

## Training
For training a model on a video, run: 
```
python train.py config/config.json
```
where the video frames folder is determined by the config parameter `"data_folder"`.
Note that in order to reduce the training time it is possible to reduce the evaluation frequency controlled by the parameter `"evaluate_every"` (e.g. by changing it to 10000).
The other configurable parameters are documented inside the file `train.py`. 

### Evaluation
During training, the model is evaluated. For running only evaluation on a trained folder run:
```
python only_evaluate.py --trained_model_folder=pretrained_models/checkpoints/blackswan --video_name=blackswan --data_folder=data --output_folder=evaluation_outputs
```
where `trained_model_folder` is the path to a folder that contains the `config.json` and `checkpoint` files of the trained model. 
## Editing
To apply editing, run the script `only_edit.py`. Examples for the supplied pretrained models for "blackswan" and "boat":
```
python only_edit.py --trained_model_folder=pretrained_models/checkpoints/blackswan --video_name=blackswan --data_folder=data --output_folder=editing_outputs --edit_foreground_path=pretrained_models/edit_inputs/blackswan/edit_blackswan_foreground.png --edit_background_path=pretrained_models/edit_inputs/blackswan/edit_blackswan_background.png
```
```
python only_edit.py --trained_model_folder=pretrained_models/checkpoints/boat --video_name=boat --data_folder=data --output_folder=editing_outputs --edit_foreground_path=pretrained_models/edit_inputs/boat/edit_boat_foreground.png --edit_background_path=pretrained_models/edit_inputs/boat/edit_boat_backgound.png
```
Where `edit_foreground_path` and `edit_background_path` specify the paths to 1000x1000 images of the RGBA atlas edits.

For applying an edit that was done on a frame (e.g. for the pretrained "libby"):

```
python only_edit.py --trained_model_folder=pretrained_models/checkpoints/libby --video_name=libby --data_folder=data --output_folder=editing_outputs  --use_edit_frame --edit_frame_index=7 --edit_frame_path=pretrained_models/edit_inputs/libby/edit_frame_.png
```

## Citation
If you find our work useful in your research, please consider citing:
```
@article{kasten2021layered,
  title={Layered neural atlases for consistent video editing},
  author={Kasten, Yoni and Ofri, Dolev and Wang, Oliver and Dekel, Tali},
  journal={ACM Transactions on Graphics (TOG)},
  volume={40},
  number={6},
  pages={1--12},
  year={2021},
  publisher={ACM New York, NY, USA}
}
```
