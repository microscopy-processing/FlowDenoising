# FlowDenoising: (Optical Flow)-driven volumetric (3D) Gaussian denoising

FlowDenoising is a Python3 module that inputs a data volume (currently [MRC](https://en.wikipedia.org/wiki/MRC_(file_format)) and [TIFF](https://en.wikipedia.org/wiki/TIFF) files are supported), removes most of the high-frequency components of the data using a Optical-Flow-driven [Gaussian kernel](https://en.wikipedia.org/wiki/Gaussian_filter) with abilities to preserve the structures and avoid blurring, and outputs the filtered volume (MRC or TIFF). The method thus reduces noise while maintaining the sharpness of the structures.The original method is described in:

> [***Structure-preserving Gaussian denoising of FIB-SEM volumes.***](https://www.sciencedirect.com/science/article/pii/S0304399122001930)  
> [V. Gonzalez-Ruiz, M.R. Fernandez-Fernandez, J.J. Fernandez.](https://www.sciencedirect.com/science/article/pii/S0304399122001930)  
> [**Ultramicroscopy** 246:113674, 2023.](https://www.sciencedirect.com/science/article/pii/S0304399122001930)  
> doi: https://doi.org/10.1016/j.ultramic.2022.113674 

Please, cite this article if you use this software in your research.

Example of use:

    > python flowdenoising.py -i vol.mrc -o denoised_vol.mrc
    
The [manual](https://github.com/microscopy-processing/FlowDenoising/blob/main/manual/manual.ipynb) provides detailed, practical information about the usage of the software.
