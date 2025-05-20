# USAGE !!
'''
python render_style_camera_original.py \
  -m output/n1statue/artistic/default \
  --style_img_path images/5.jpg \
  --target_style_img_path images/0.jpg \
  --steps 121
  --interp_per_cam 5
'''
'''
ffmpeg -framerate 24 -i ./output/n1statue/artistic/default/camera_style_interpolation_5_0/%04d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p output.mp4
'''

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

# Add for bad images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
import os
from pathlib import Path

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from scene.VGG import VGGEncoder, normalize_vgg
from scene.cameras import Camera
from argparse import ArgumentParser



def render_sets_style_camera_interpolate(dataset: ModelParams, pipeline: PipelineParams, style_img_path, steps=10, interp_per_cam=5, target_style_img_path=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        ckpt_path = os.path.join(dataset.model_path, "chkpnt/gaussians.pth")
        scene = Scene(dataset, gaussians, load_path=ckpt_path, shuffle=False, style_model=True)

        trans = T.Compose([T.Resize(size=(256, 256)), T.ToTensor()])
        style_img = trans(Image.open(style_img_path)).cuda()[None, :3, :, :]
        style_index = Path(style_img_path).stem

        vgg_encoder = VGGEncoder().cuda()
        style_img_features = vgg_encoder(normalize_vgg(style_img))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if target_style_img_path is not None:
            target_style_img = trans(Image.open(target_style_img_path)).cuda()[None, :3, :, :]
            target_index = Path(target_style_img_path).stem
            target_style_features = vgg_encoder(normalize_vgg(target_style_img))
            start_feature = gaussians.style_transfer(gaussians.final_vgg_features.detach(), style_img_features.relu3_1)
            end_feature = gaussians.style_transfer(gaussians.final_vgg_features.detach(), target_style_features.relu3_1)
        else:
            target_index = "base"
            start_feature = gaussians.final_vgg_features
            end_feature = gaussians.style_transfer(gaussians.final_vgg_features.detach(), style_img_features.relu3_1)

        render_path = os.path.join(dataset.model_path, f"camera_style_interpolation_{style_index}_{target_index}")
        os.makedirs(render_path, exist_ok=True)

        cameras = scene.getTrainCameras()
        cam_count = len(cameras)
        total_frames = (cam_count - 1) * interp_per_cam

        for i in tqdm(range(total_frames), desc="Rendering camera+style interpolation"):
            cam_idx = i // interp_per_cam
            alpha = (i % interp_per_cam) / interp_per_cam

            cam0 = cameras[cam_idx]
            cam1 = cameras[cam_idx + 1]

            R0 = torch.tensor(cam0.R, device="cuda")
            T0 = torch.tensor(cam0.T, device="cuda")
            R1 = torch.tensor(cam1.R, device="cuda")
            T1 = torch.tensor(cam1.T, device="cuda")

            R_interp = (1 - alpha) * R0 + alpha * R1
            T_interp = (1 - alpha) * T0 + alpha * T1

            style_alpha = i / total_frames
            features_interp = (1 - style_alpha) * start_feature + style_alpha * end_feature
            override_color = gaussians.decoder(features_interp)

            cam_clone = Camera(
                colmap_id=None,
                R=R_interp.cpu().numpy(),
                T=T_interp.cpu().numpy(),
                FoVx=cam0.FoVx,
                FoVy=cam0.FoVy,
                image=None,
                gt_alpha_mask=None,
                image_name=None,
                uid=None
            )
            cam_clone.image_height = cam0.image_height
            cam_clone.image_width = cam0.image_width

            rendering = render(cam_clone, gaussians, pipeline, background, override_color=override_color)["render"]
            rendering = rendering.clamp(0, 1)

            save_path = os.path.join(render_path, f'{i:04d}.png')
            torchvision.utils.save_image(rendering, save_path)




if __name__ == "__main__":
    parser = ArgumentParser(description="Render with style and camera interpolation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--style_img_path", type=str, required=True, help="Path to style image")
    parser.add_argument("--view_id_start", type=int, default=0, help="Start camera id")
    parser.add_argument("--view_id_end", type=int, default=10, help="End camera id")
    parser.add_argument("--steps", type=int, default=10, help="Number of interpolation steps")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    parser.add_argument("--target_style_img_path", type=str, default=None, help="Optional: Path to target style image")
    parser.add_argument("--interp_per_cam", type=int, default=5, help="Number of interpolation steps between each camera pair")


    args = get_combined_args(parser)

    print(f"Rendering with style {args.style_img_path} from view {args.view_id_start} to {args.view_id_end} with {args.steps} steps")

    safe_state(args.quiet)

    render_sets_style_camera_interpolate(
    dataset=model.extract(args),
    pipeline=pipeline.extract(args),
    style_img_path=args.style_img_path,
    steps=args.steps,
    interp_per_cam=args.interp_per_cam,
    target_style_img_path=args.target_style_img_path
    )
