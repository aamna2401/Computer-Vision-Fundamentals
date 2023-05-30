#!/usr/bin/env bash
# Regenerate every figure and metrics.json. Topic 9 is skipped unless FACES=/path/to/celeba_crops is set.
set -e
cd "$(dirname "$0")"
for d in 01_canny_edge_detection 02_harris_corners 03_bag_of_visual_words 04_lane_detection \
         05_camera_calibration_epipolar 06_image_rectification 07_classical_cv_practice 08_cnn_from_scratch; do
  echo "== $d"; (cd "$d" && python demo.py)
done
echo "== 10_graph_convolution_networks"
(cd 10_graph_convolution_networks && python demo_cora.py && python demo_skeleton.py)
if [ -n "$FACES" ]; then echo "== 09"; (cd 09_face_recognition_celeba && python demo.py --data "$FACES"); fi
