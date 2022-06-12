# FlowDenoising: (Optical Flow (OF))-driven volumetric (3D) Gaussian denoising

FlowDenoising is a Python3 module that inputs a data volume (currently [MRC](https://en.wikipedia.org/wiki/MRC_(file_format)) and [TIFF](https://en.wikipedia.org/wiki/TIFF) files are supported), low-pass filters the data using a OF-driven [Gaussian kernel](https://en.wikipedia.org/wiki/Gaussian_filter), and outputs the filtered volume (MRC or TIFF).

Example of use:

    > python flowdenoising.py --input empiar10311_crop.mrc --output filtered_empiar10311_crop.mrc
    
Please, read the [manual](https://github.com/microscopy-processing/FlowDenoising/blob/main/manual/manual.ipynb) for getting more information.
