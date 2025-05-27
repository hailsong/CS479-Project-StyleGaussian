#!/bin/bash

# Set model directory
MODEL_PATH="output/saengsaeng/artistic/default"

# Style interpolations
python render_style_camera_original.py -m $MODEL_PATH --style_img_path images/22.jpg --target_style_img_path images/43.jpg --steps 121 --interp_per_cam 5
python render_style_camera_original.py -m $MODEL_PATH --style_img_path images/15.jpg --target_style_img_path images/43.jpg --steps 121 --interp_per_cam 5
python render_style_camera_original.py -m $MODEL_PATH --style_img_path images/42.jpg --target_style_img_path images/45.jpg --steps 121 --interp_per_cam 5

# Video generation
ffmpeg -framerate 60 -i ./output/saengsaeng/artistic/default/camera_style_interpolation_22_43/%04d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p output_22_43.mp4
ffmpeg -framerate 60 -i ./output/saengsaeng/artistic/default/camera_style_interpolation_15_43/%04d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p output_15_43.mp4
ffmpeg -framerate 60 -i ./output/saengsaeng/artistic/default/camera_style_interpolation_42_45/%04d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p output_42_45.mp4
