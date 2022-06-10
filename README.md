# FlowDenoising (Optical flow - driven volumetric (3D) Gaussian denoising)

FlowDenoising inputs a volume (MRC and TIFF files are currently accepted), low-pass filters the volume, and outputs a volume (a MRC file or a sequence of TIFF files).

Example:
    > ls -l *.mrc
    empiar10311_crop.mrc
    > python flowdenoising.py --sigma 2.0 --input empiar10311_crop.mrc --output filtered_empiar10311_crop.mrc
    > ls -l *.mrc
    empiar10311_crop.mrc
    filtered_empiar10311_crop.mrc
    
In the [manual](https://github.com/microscopy-processing/FlowDenoising/blob/main/manual/manual.ipynb) you will find more information.
