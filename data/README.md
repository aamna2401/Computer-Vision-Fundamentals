# Sample data

| folder | contents | used by |
|---|---|---|
| `images/` | checkerboards, synthetic shapes, a Lahore fort facade, a book cover, a classic test portrait, Haar face cascade | 01, 02, 07 |
| `lane_frames/` | six dashcam frames (highway driving) | 04 |
| `calibration/` | 20 known 2-D/3-D point pairs; a temple stereo pair with 110 matches and camera matrices | 05 |
| `rectification/` | a perspective-distorted tile floor | 06 |
| `scenes/train/` | 15 scene classes x 40 images (subset of the public 15-Scene dataset, Lazebnik et al.) | 03 |
| `video/` | 4-second 480x270 street clip | 07 |
| `cora/` | Cora citation graph, raw files (Planetoid) | 10 |

Faces (topic 9) are not redistributed: CelebA is licensed for non-commercial research use. Obtain it from its authors and point `demo.py --data` at a folder of `<identity>/*.jpg`.

Check the licence of any image before reusing it outside this repository; several come from course material and public datasets.
