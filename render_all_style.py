'''
python render_all_style.py \
  -m output/saengsaeng/artistic/default \
  --style_img_path images/5.jpg \
  --target_style_img_path images/0.jpg \
  --interp_per_cam 5 \
  --save_every 10 \
  --frame_max 130

python render_all_style.py \
  -m output/saengsaeng/artistic/default \
  --style_img_path images/ \
  --interp_per_cam 5 \
  --save_every 10 \
  --frame_max 130 \
  --target_style_img_path images/ \
  --quiet

python render_all_style.py \
  -m output/saengsaeng/artistic/default \
  --style_dir images/ \
  --interp_per_cam 5
'''

# render_all_style.py
import torch, torchvision
import torchvision.transforms as tvT
from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from pathlib import Path
import os, glob
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from scene.VGG import VGGEncoder, normalize_vgg
from scene.cameras import Camera
from argparse import ArgumentParser


def render_style_batch(dataset, pipeline, style_img, interp_per_cam=5):
    """렌더는 0~130 프레임 중 10의 배수만 저장"""
    with torch.no_grad():
        # --- scene & model --------------------------------------------------
        gaussians = GaussianModel(dataset.sh_degree)
        ckpt      = os.path.join(dataset.model_path, "chkpnt/gaussians.pth")
        scene     = Scene(dataset, gaussians, load_path=ckpt,
                          shuffle=False, style_model=True)

        # --- style feature --------------------------------------------------
        trans      = tvT.Compose([tvT.Resize((256, 256)), tvT.ToTensor()])
        img_tensor = trans(Image.open(style_img)).cuda()[:3].unsqueeze(0)
        vgg        = VGGEncoder().cuda()
        style_feat = vgg(normalize_vgg(img_tensor))
        base_feat  = gaussians.final_vgg_features
        stylised   = gaussians.style_transfer(base_feat.detach(),
                                              style_feat.relu3_1)

        # --- output dir -----------------------------------------------------
        stem        = Path(style_img).stem
        out_dir     = os.path.join(dataset.model_path,
                                   f"camera_style_{stem}")
        os.makedirs(out_dir, exist_ok=True)

        # --- camera list & frame indices ------------------------------------
        cams         = scene.getTrainCameras()
        tot_frames   = (len(cams) - 1) * interp_per_cam
        frame_idx    = [f for f in range(0, min(131, tot_frames), 10)]

        bg           = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background   = torch.tensor(bg, dtype=torch.float32, device="cuda")

        # --- rendering loop --------------------------------------------------
        for i in tqdm(frame_idx, desc=f"[{stem}]"):
            c0, c1 = cams[i // interp_per_cam], cams[i // interp_per_cam + 1]
            a      = (i % interp_per_cam) / interp_per_cam

            R0, T0 = torch.tensor(c0.R, device="cuda"), torch.tensor(c0.T, device="cuda")
            R1, T1 = torch.tensor(c1.R, device="cuda"), torch.tensor(c1.T, device="cuda")
            Rm     = (1 - a) * R0 + a * R1
            Tm     = (1 - a) * T0 + a * T1

            cam_i = Camera(None, Rm.cpu().numpy(), Tm.cpu().numpy(),
                           c0.FoVx, c0.FoVy, None, None, None, None)
            cam_i.image_height, cam_i.image_width = c0.image_height, c0.image_width

            img = render(cam_i, gaussians, pipeline,
                         background, override_color=gaussians.decoder(stylised)
                         )["render"].clamp(0, 1)
            torchvision.utils.save_image(img, f"{out_dir}/{i:04d}.png")


def collect_images(directory):
    exts = ("*.png", "*.jpg", "*.jpeg")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(directory, e)))
    return sorted(paths)


if __name__ == "__main__":
    p = ArgumentParser("Batch-style renderer (every 10 frames)")
    model    = ModelParams(p, sentinel=True)   # <-- 이미 -m 추가됨
    pipeline = PipelineParams(p)
    p.add_argument("--style_dir", type=str, required=True,
                   help="Directory with style images")
    p.add_argument("--interp_per_cam", type=int, default=5)
    p.add_argument("--quiet", action="store_true")
    args = get_combined_args(p)
    safe_state(args.quiet)

    for s_img in collect_images(args.style_dir):
        render_style_batch(
            dataset=model.extract(args),
            pipeline=pipeline.extract(args),
            style_img=s_img,
            interp_per_cam=args.interp_per_cam
        )
