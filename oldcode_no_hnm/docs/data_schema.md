Project Data Schema Definition

This document defines the standard data format for all processed data to ensure consistency between scripts.

1. pose_npz (Interim)

Location: data/interim/<dataset>/pose_npz/*.npz

This is the direct output of pose/extract_2d.py. Each file contains the full pose data for one video.

xy: np.array[T, 33, 2] (float32, np.nan for no-detect)

conf: np.array[T, 33] (float32, 0.0 for no-detect)

T = Total frames in the video

J = 33 (BlazePose keypoints)

2. window_npz (Processed)

Location: data/processed/<dataset>/windows_W48_S12/*.npz

This is the final, model-ready data, created by pose/make_windows.py. Each file is one window.

xy: np.array[T, 33, 2] (float32)

conf: np.array[T, 33] (float32)

start: (int) The start frame index (from original video)

end: (int) The end frame index

label: (str) e.g., "fall" or "adl"

video_id: (str) The source video identifier