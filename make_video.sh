#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <subject_name> <style_index_start> <style_index_end>"
    echo "Example: $0 saengsaeng 22 43"
    exit 1
fi

subject="$1"
start="$2"
end="$3"

interp_dir="./output/${subject}/artistic/default/camera_style_interpolation_${start}_${end}"
tmp_dir="tmp_frames"

mkdir -p "$tmp_dir"

cp "$interp_dir"/*.png "$tmp_dir/"

frames=($interp_dir/*.png)
N=${#frames[@]}

idx=$N
for ((i=N-1; i>=0; i-=2)); do
    f="${frames[i]}"
    printf -v newname "%04d.png" "$idx"
    cp "$f" "$tmp_dir/$newname"
    idx=$((idx + 1))
done

ffmpeg -framerate 120 -i "$tmp_dir/%04d.png" \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p "looped_${subject}_${start}_${end}.mp4"

rm -r "$tmp_dir"
