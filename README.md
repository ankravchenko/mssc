Computes multi-scale structural complexity of images in a dataset using a variety of coarse graining methods.

Currently implemented:

    Fourier Transform
    Gaussian blur
    Rank filter (median)
    Wavelet transform, python 2d wavelets (py_cwt2d)

The main experiment script (complexity_colour_fft.py) uses coarse graining implementation based on Fourier Transform, taking two input parameters: path to the folder contianing images, path to a file with human complexity rankings It can be started with: python complexity_colour_fft.py scenes "dataset_folder" "ranking_file.xlsx" fft A *.csv file with resulting MSSC rankings is stored in the "calculated_mssc_100/" folder. Scatter plots and regression analysis are stored in the "mssc_figures_100/" folder.

complexity_colour_fft.sh can be used for batch computing MSSC for the entirety of SAVOIAS dataset, by default the dataset is implied to be stored in the ../Savoias-Dataset folder.

Scripts scale_search.py and plot_scale_impact.py can be used for plotting the impact of scales on the overall complexity rankings and correlation of each scale ranking with human predictions for a given subset of SAVOIAS. scale_search.sh and plot_scale_impact.sh plot these for all subsets in SAVOIAS set.

Conda environment config containing all required packages is stored in mssc.yml

Scripts complexity_rank_colour.py, complexity_colour_blur.py and complexity_colour_wavelets.py can be used for alternative methods of coarse graining.

Questions can be addressed to anna.kravchenko@ru.nl

Please reference: Kravchenko, A., Bagrov, A. A., Katsnelson, M. I., & Dudarev, V. (2024). Multi-scale structural complexity as a quantitative measure of visual complexity. arXiv preprint arXiv:2408.04076.

