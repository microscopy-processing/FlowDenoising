# FlowDenoising: (Optical Flow)-driven volumetric (3D) Gaussian denoising

FlowDenoising inputs a data volume (currently MRC and TIFF files are accepted), low-pass filters the data using a Gaussian kernel, and outputs the filtered volume (a MRC file or a sequence of TIFF files).

Example:

    > ls -l *.mrc
    empiar10311_crop.mrc
    > python flowdenoising.py --sigma 2.0 --input empiar10311_crop.mrc --output filtered_empiar10311_crop.mrc
    > ls -l *.mrc
    empiar10311_crop.mrc
    filtered_empiar10311_crop.mrc
    
Please, read the [manual](https://github.com/microscopy-processing/FlowDenoising/blob/main/manual/manual.ipynb) for more information.
