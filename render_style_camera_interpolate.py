# USAGE !!
# python render_style_camera_interpolate.py -m output/Family/artistic/default --style_img_path images/van_gogh.png --view_id_start 0 --view_id_end 10 --steps 20
# ffmpeg -framerate 24 -i ./output/test_images/artistic/default/camera_style_interpolation_2/%04d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p output.mp4


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

def render_sets_style_camera_interpolate(dataset : ModelParams, pipeline : PipelineParams, style_img_path, view_id_start=0, view_id_end=10, steps=10):
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

        # get style transfered features
        tranfered_features = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(),
            style_img_features.relu3_1,
        )

        # get cameras
        view_start = scene.getTrainCameras()[view_id_start]
        view_end = scene.getTrainCameras()[view_id_end]

        R_start = torch.from_numpy(view_start.R).float().cuda()
        T_start = torch.from_numpy(view_start.T).float().cuda()
        R_end = torch.from_numpy(view_end.R).float().cuda()
        T_end = torch.from_numpy(view_end.T).float().cuda()

        v = torch.linspace(0, 1, steps=steps)

        for i in tqdm(range(steps), desc="Rendering interpolated views"):
            alpha = v[i]

            # interpolate rotation and translation
            R_interp = (1 - alpha) * R_start + alpha * R_end
            T_interp = (1 - alpha) * T_start + alpha * T_end

            # normalize rotation matrix (simple method: Gram-Schmidt orthogonalization)
            u, _, vh = torch.linalg.svd(R_interp)
            R_interp = u @ vh

            # interpolate features
            features_interp = (1 - alpha) * gaussians.final_vgg_features + alpha * tranfered_features
            override_color = gaussians.decoder(features_interp)

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
    args = get_combined_args(parser)

    print(f"Rendering with style {args.style_img_path} from view {args.view_id_start} to {args.view_id_end} with {args.steps} steps")

    safe_state(args.quiet)

    render_sets_style_camera_interpolate(
        dataset=model.extract(args),
        pipeline=pipeline.extract(args),
        style_img_path=args.style_img_path,
        view_id_start=args.view_id_start,
        view_id_end=args.view_id_end,
        steps=args.steps
    )
