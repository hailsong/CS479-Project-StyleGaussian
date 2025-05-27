python train.py --data datasets/test_images --wikiartdir wiki_datasets/ --exp_name default

# reconstruct : 07:44

# feature : 04:00

# artistic : 엄청 오래걸릴듯듯
# python train_artistic.py -s datasets/test_images/ --wikiartdir wiki_datasets/2/ --ckpt_path output/test_images/feature/default/chkpnt/feature.pth --style_weight 10

# === Process Time Summary ===
# Reconstruction      : 16m 55.5s
# Feature Embedding   : 8m 18.7s
# Style Transfer      : 439m 46.4s

% Rendering
python render_style_camera_interpolate.py -m output/saengsaeng/artistic/default --style_img_path images/22.jpg --target_style_img_path images/43.jpg --view_id_start 4 --view_id_end 138 --steps 150