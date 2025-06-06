# USAGE !!
# python render_style_camera_interpolate.py -m output/test_images/artistic/default --style_img_path images/1.jpg --target_style_img_path images/2.jpg --view_id_start 0 --view_id_end 10 --steps 150
# ffmpeg -framerate 24 -i ./output/test_images/artistic/default/camera_style_interpolation_3/%04d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p output.mp4


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



def render_sets_style_camera_interpolate(dataset : ModelParams, pipeline : PipelineParams, style_img_path, view_id_start=0, view_id_end=10, steps=10, target_style_img_path=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        ckpt_path = os.path.join(dataset.model_path, "chkpnt/gaussians.pth")
        scene = Scene(dataset, gaussians, load_path=ckpt_path, shuffle=False, style_model=True)

        # read style image
        trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])
        style_img = trans(Image.open(style_img_path)).cuda()[None, :3, :, :]
        style_name = Path(style_img_path).stem
        vgg_encoder = VGGEncoder().cuda()
        style_img_features = vgg_encoder(normalize_vgg(style_img))

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(dataset.model_path, f"camera_style_interpolation_{style_name}")
        os.makedirs(render_path, exist_ok=True)

        # Style transfer
        transfered_features = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(),
            style_img_features.relu3_1,
        )

        # Optional: target style
        if target_style_img_path is not None:
            target_style_img = trans(Image.open(target_style_img_path)).cuda()[None, :3, :, :]
            target_style_features = vgg_encoder(normalize_vgg(target_style_img))
            target_transfered_features = gaussians.style_transfer(
                gaussians.final_vgg_features.detach(),
                target_style_features.relu3_1,
            )
            start_feature = transfered_features
            end_feature = target_transfered_features
        else:
            start_feature = gaussians.final_vgg_features
            end_feature = transfered_features

        # ---- 이하 나머지는 interpolation만 수정 ----
        view_start = scene.getTrainCameras()[view_id_start]
        view_end = scene.getTrainCameras()[view_id_end]

        R_start = torch.tensor(view_start.R, device="cuda")
        T_start = torch.tensor(view_start.T, device="cuda")
        R_end = torch.tensor(view_end.R, device="cuda")
        T_end = torch.tensor(view_end.T, device="cuda")

        v = torch.linspace(0, 1, steps=steps)
        v = torch.linspace(0, 1, steps=steps)

        for i in tqdm(range(steps), desc="Rendering interpolated views"):
            alpha = v[i]

            # interpolate features
            features_interp = (1 - alpha) * start_feature + alpha * end_feature
            override_color = gaussians.decoder(features_interp)

            R_interp = (1 - alpha) * R_start + alpha * R_end
            T_interp = (1 - alpha) * T_start + alpha * T_end

            # build virtual Camera
            cam = Camera(
                colmap_id=None,
                R=R_interp.cpu().numpy(),
                T=T_interp.cpu().numpy(),
                FoVx=view_start.FoVx,
                FoVy=view_start.FoVy,
                image=None,
                gt_alpha_mask=None,
                image_name=None,
                uid=None
            )
            cam.image_height = view_start.image_height
            cam.image_width = view_start.image_width

            # render
            rendering = render(cam, gaussians, pipeline, background, override_color=override_color)["render"]
            rendering = rendering.clamp(0, 1)

            torchvision.utils.save_image(rendering, os.path.join(render_path, f'{i:04d}.png'))


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

    args = get_combined_args(parser)

    print(f"Rendering with style {args.style_img_path} from view {args.view_id_start} to {args.view_id_end} with {args.steps} steps")

    safe_state(args.quiet)

    render_sets_style_camera_interpolate(
    dataset=model.extract(args),
    pipeline=pipeline.extract(args),
    style_img_path=args.style_img_path,
    view_id_start=args.view_id_start,
    view_id_end=args.view_id_end,
    steps=args.steps,
    target_style_img_path=args.target_style_img_path
    )
