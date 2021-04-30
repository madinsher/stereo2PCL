
python single_camera_calibration.py \
    --image_dir ./data/rightcamera \
    --image_format jpg \
    --prefix right \
    --square_size 0.04 \
    --width 9 \
    --height 12  \
    --save_file ./configs/right_cam.yml


python single_camera_calibration.py \
    --image_dir ./data/ \
    --image_format jpg \
    --prefix left \
    --square_size 0.04 \
    --width 9 --height 12 \
    --save_file ./configsleft_cam.yml