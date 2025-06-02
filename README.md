
# [2025 CS479 Rendering Contest] 3SGS : SaengSaeng Stylized Gaussian Splatting

Implementation of StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting (SIGGRAPH Asia 2024) as part of the CS479 Rendering Contest.

This project enables real-time 3D style transfer using 3D Gaussian Splatting and style images from WikiArt. It supports both style transfer and interpolation across multiple viewpoints.

For the details, please refer the (project report)[./project_report.pdf]

Pretrained model can be found at : [pretrained model](https://drive.google.com/file/d/1f7xMMzEPfS3Su91_fFh4dkioDjkIx7cn/view?usp=drive_link)


For rendering-only usage, please refer to the following sections:

- 0 Environment Setup
- 1.1 Pretrained Model
- 3 Interactive Viewer
------

https://github.com/user-attachments/assets/f497e90c-4dd8-4deb-9874-da70d99043cb




## 0. Environment Setup

We recommend using `mamba` or `conda` to set up the environment:

```
mamba env create -f environment.yml -n stylegaussian
conda activate stylegaussian
```

------

## 1. Project Structure & Setup
### 1.1 Pretrained Model
Download the pretrained model from Google Drive and extract it to:

```
CS479-Project-StyleGaussian/output/saengsaeng_pretrained
└── artistic
    └── default
        ├── cfg_args/
        └── chkpnt/
            └── gaussians.pth       # Pretrained Gaussian model checkpoint
```

### 1.2 Style Images (Optional, for training)
We use style images from the [WikiArt Dataset](https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan) as the style images dataset.
To use your own styles, simply place your chosen images in the ./images/ directory.

------

## 2. Quick Start
### 2.1 Basic Inference

```
python main.py \
  -m ./output/<MODEL_NAME>/artistic/default \
  --style_img_path <STYLE_IMAGE_1> \
  --target_style_img_path <STYLE_IMAGE_2>
```

- **Example:**
```
python main.py \
  -m ./output/saengsaeng_pretrained/artistic/default \
  --style_img_path images/22.jpg \
  --target_style_img_path images/43.jpg
```

This code will automatically render all images for the video.  
After generating the images, use `make_video.sh` to produce the final looping video.


### 2.2 Generate Video Output

- After rendering the images, run the following script to generate a looping video:

```
# Usage:
bash make_video.sh <MODEL_NAME> <STYLE_IMAGE_1> <STYLE_IMAGE_2>
```

- **Example:**

```
bash make_video.sh saengsaeng 22 43
```
This will output a stylized looping video in the `videos/` folder, combining all rendered frames.



---

## 3. Interactive Web Viewer

We provide an interactive real-time **style transfer viewer** using the [Viser](https://github.com/nerfstudio-project/viser) interface. This allows you to explore multi-view 3D stylization results directly in your web browser.

To launch the viewer:

```
bash
python viewer.py -m output/saengsaeng/artistic/default --style_folder images --viewer_port 8080
```
This command starts a web viewer at:
http://localhost:8080 (or http://<your-ip>:8080 if accessed over a LAN or tunnel).

💡 If you're running this on a remote server, make sure to forward or expose the correct port. For local networks, access http://<host-machine-IP>:8080 from another device.

🔧 Viewer Features
- Rendering Settings
- Style Transfer
- Angle-Based Style Transfer (Same with our project video output)


---

## 4. About the Original Paper

https://github.com/Kunhao-Liu/StyleGaussian/assets/63272695/d6dfda95-b272-42ff-b855-e16801f594a9

This project is based on [**StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting (SIGGRAPH Asia 2024)**](https://arxiv.org/abs/2403.07807).

StyleGaussian proposes a real-time 3D style transfer pipeline using 3D Gaussian Splatting, achieving multi-view consistency and fast rendering.

- [Original Paper (arXiv)](https://arxiv.org/abs/2403.07807)
- [Project Page](https://kunhao-liu.github.io/StyleGaussian/)
- [Official Codebase](https://github.com/Kunhao-Liu/StyleGaussian)

In our CS479 Rendering Contest project, we utilized the pretrained model and simplified the pipeline to focus on **style transfer inference and stylized video generation**.


### 4.1 Installation (from Original Repo)

To set up the environment, we follow the original repository’s setup:

```
mamba env create -f environment.yml -n stylegaussian
conda activate stylegaussian
```
Alternatively, you can use conda if mamba is not available.


### 4.2 Quick Start
[Datasets and Checkpoints (Google Drive)](https://drive.google.com/drive/folders/1xHGXniVL3nh6G7pKDkZR1SJlfvo4YB1J?usp=sharing)
Please download the pre-processed datasets and put them in the `datasets` folder. We also provide the pre-trained checkpoints, which should be put in the `output` folder. If you change the location of the datasets or the location of the checkpoints, please modify the `model_path` or `source_path` accordingly in the `cfg_args` in the checkpoint folder.


#### 4.2.1 Interactive Remote Viewer

The original StyleGaussian repository provides an interactive web-based viewer based on [Viser](https://github.com/nerfstudio-project/viser). 

- To launch the viewer:

```bash
python viewer.py -m [model_path] --style_folder [style_folder] --viewer_port [viewer_port]
```

- Example:
```
python viewer.py -m output/train/artistic/default --style_folder images --viewer_port 8080
```

where `model_path` is the path to the pre-trained model, named as `output/[scene_name]/artistic/[exp_name]`, `style_folder` is the folder containing the style images, and `viewer_port` is the port number for the viewer. `--style_folder` and `--viewer_port` are optional arguments.


#### 4.2.2 Inference Rendering

The original repository provides inference rendering using `render.py` with either a single style image or a set of four images for style interpolation.

- **Single Style Transfer:**
```
python render.py -m [model_path] --style [style_image_path]
```

- **Example:**
```
python render.py -m output/train/artistic/default --style images/0.jpg
```

- **Style Interpolation (4 images):**

```
python render.py -m [model_path] --style image1.jpg image2.jpg image3.jpg image4.jpg
```

- **The rendered images are saved in:**
```
output/[scene_name]/artistic/[exp_name]/train/
```
```
# or
.../style_interpolation/
```


### 4.3 Acknowledgements

Our work is based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [StyleRF](https://github.com/Kunhao-Liu/StyleRF). We thank the authors for their great work and open-sourcing the code.



### 4.4 Citation

```
@article{liu2023stylegaussian,
  title={StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting},
  author={Liu, Kunhao and Zhan, Fangneng and Xu, Muyu and Theobalt, Christian and Shao, Ling and Lu, Shijian},
  journal={arXiv preprint arXiv:2403.07807},
  year={2024},
}
```
