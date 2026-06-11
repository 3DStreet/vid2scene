# Direct .insv dual-fisheye processing

vid2scene can process Insta360 `.insv` recordings directly from their two raw
fisheye sensor streams, without first stitching them to an equirectangular
video in Insta360 Studio.

## Why

The equirectangular (ER) path loses exactly the part of the scene users
notice most: the ground.

1. **Stitching resamples the sphere.** An ER image stretches the poles
   (straight down and straight up) across the entire bottom/top row of
   pixels. The ground directly under the camera — already the noisiest
   region optically — is smeared before SfM ever sees it.
2. **The ER virtual rig never looks down.** `pano_sfm.py` renders 12 virtual
   views at pitches `(-35°, 0°, +35°)` with 90° FOV, so nothing below
   −80° pitch is covered at all. The nadir gets zero direct observations.

The fisheye-direct path (`fisheye_sfm.py`) crops virtual pinhole views
straight out of each fisheye stream:

- Native sensor pixels everywhere. An equidistant fisheye has roughly
  uniform angular resolution; the nadir sits ~90° off each lens axis, well
  inside a ~200° lens, at full resolution.
- The default view grid per lens is 3 yaws × 3 pitches at ±60° with 75°
  square crops (9 views per lens, **18 per frame pair**). The down-pitched
  views of both lenses each put the nadir ~30° from their optical axis, so
  the ground under the camera is covered by **up to 6 views per frame pair**
  instead of 0.
- All 18 views are registered as a single COLMAP rig (shared orientation,
  zero translation), same machinery as the ER path; the SfM steps
  (`pano_sfm.run_rig_sfm_pipeline`) are shared.

## Usage

```bash
# Auto-detected from the .insv extension:
python vid2scene_core/vid2scene.py output_dir --video_path VID_20260611_120000_00_001.insv

# Explicit, with a lens FOV override:
python vid2scene_core/vid2scene.py output_dir --video_path capture.insv \
    --insv_fisheye --insv_lens_fov 195

# Run only the SfM stage:
python vid2scene_core/fisheye_sfm.py --insv_path capture.insv --output_path out/
```

Supported `.insv` layouts (demuxed with ffmpeg):

| Layout | Cameras | Handling |
|---|---|---|
| Two video streams in one file | X3 / X4 / X5 single-file recordings | streams `0:v:0` and `0:v:1` |
| Two files `..._00_...insv` + `..._10_...insv` | older cameras, some high-res modes | pass either file; the companion is found automatically |
| One 2:1 stream, two fisheyes side by side | some older models | split in half; guarded by a dark-corner check so ER videos aren't mis-split |

## Lens model and calibration

By default each lens is modeled as an idealized equidistant fisheye:
principal point at the frame center, image circle inscribed in the frame,
and a nominal 200° FOV mapped linearly onto the circle
(`fisheye_projection.FisheyeLensModel`). Real lenses deviate; if the
reconstruction shows systematic warp, supply a calibration JSON via
`--insv_calibration`:

```json
{
  "fov_deg": 197.5,
  "k1": -0.012, "k2": 0.0021, "k3": 0.0, "k4": 0.0,
  "lenses": [
    {"cx": 1441.2, "cy": 1439.1, "focal_px_per_rad": 824.5},
    {"cx": 1438.9, "cy": 1442.3, "focal_px_per_rad": 826.1}
  ],
  "lens1_yaw_deg": 180.0,
  "lens1_roll_deg": 0.0
}
```

Top-level keys apply to both lenses; entries in `lenses` override per lens.
`k1..k4` are odd-polynomial (Kannala-Brandt) distortion terms on
`r = focal * (θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹)`. `lens1_yaw_deg` /
`lens1_roll_deg` describe how the rear lens is mounted relative to the front
one.

## Masks

For every rendered view two masks are produced:

- **SfM masks** (`masks/`): valid-fisheye-pixel ∩ closest-view partition,
  so overlapping views don't generate duplicate SIFT features of the same
  physical point. COLMAP requires a mask for every image once `mask_path`
  is set; one is written per rendered image.
- **Training masks** (`training_masks/`): valid fisheye pixels only, so
  gsplat doesn't train on the black corners of views that extend past the
  lens FOV.

## Current limitations / follow-ups

- **No per-unit factory calibration.** Insta360 ships MEI-model calibration
  (protobuf, in the trailer or an `.insv.pb` sidecar on X5); parsing it
  would remove the idealized-lens assumption. The
  [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser) crate
  (used by [insv-stitch](https://github.com/BenjaminHenriksson/insv-stitch))
  already reads these.
- **No IMU use.** The trailer's gyro record (`0x300`) could provide
  horizon leveling and rolling-shutter correction.
- **Lens baseline ignored.** The two lens centers are ~2–3 cm apart but the
  rig assumes a shared center, like the ER path (and Insta360's own
  stitching). Only matters for subjects very close to the camera.
- **No ego-object masking.** The ER path can mask the operator/tripod with
  SAM3; that step is not yet wired into the fisheye path.
- **Trailer parsing is best-effort.** Based on community reverse
  engineering (ExifTool, Sub-Etha Software); used only for logging camera
  metadata. Failures are silent and harmless.
- **Server/API wiring.** The web upload flow only exposes the
  `equirectangular` flag; `.insv` uploads through the server need a model
  field + form pass-through (auto-detection by extension already works at
  the pipeline level).
