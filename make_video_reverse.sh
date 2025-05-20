mkdir -p tmp_frames

# 정방향 복사
cp ./output/n1statue/artistic/default/camera_style_interpolation_5_0/*.png tmp_frames/

# 프레임 수 계산
N=$(ls ./output/n1statue/artistic/default/camera_style_interpolation_5_0/*.png | wc -l)

# 역방향 복사하면서 이름 새로 부여
idx=$N
for f in $(ls ./output/n1statue/artistic/default/camera_style_interpolation_5_0/*.png | sort -r); do
    printf -v newname "%04d.png" "$idx"
    cp "$f" tmp_frames/"$newname"
    idx=$((idx + 1))
done

# 영상 렌더링
ffmpeg -framerate 60 -i tmp_frames/%04d.png \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p looped_output.mp4

# 필요 시 정리
rm -r tmp_frames
