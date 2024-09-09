Computes multi-scale structural complexity of images in a dataset using coarse graining implementation based on Fourier Transform.

The main experiment script (complexity_colour_fft.py) takes three input parameters: 
- path to the folder contianing images, 
- path to a file with human complexity rankings
- coarse graining type (currently only 'fft' can be used)

Example:
python complexity_colour_fft.py scenes dataset_folder ranking_file.xlsx fft 

A *.csv file with resulting partial complexities will be stored in the results/calculated_mssc/" folder. Scatter plots and regression analysis will be stored in the results/mssc_figures/ folder. The poarse graining process visualisations will be stored in the "image_detail/" folder.
This file doesn't yet contain the correct MSSC values, only intermediate results.

middle_scale.py dataset_name calculates MSSC cutting off high and low spatial frequencies (middle scales of the image have been shown to have the highest correlation with human participants), storing the results in results/fft_datasetname_mssc.csv.
**MSSC column in this file is the final multi-scale structural complexity value.**
The same script also calculates correlation with subjective rankings and plots regression graphs.

Scripts scale_search.py and plot_scale_impact.py can be used for plotting the impact of scales on the overall complexity rankings and correlation of each scale ranking with human predictions for a given subset of SAVOIAS. 

Conda environment config containing all required packages is stored in mssc.yml

To reproduce our experiment and calculate MSSC and correlations for the entirety of SAVOIAS dataset (by default it's implied to be stored in the ../Savoias-Dataset folder) run:

mv "../Savoias-Dataset/Images/Art/global_ranking/" "../Savoias-Dataset/"\\
complexity_colour_fft.sh\\
middle_scale.sh \\
plot_scale_impact.sh \\


