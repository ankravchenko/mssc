Computes multi-scale structural complexity of images in a dataset using coarse graining implementation based on Fourier Transform.

The main experiment script (complexity_colour_fft.py) takes three input parameters: 
- path to the folder contianing images, 
- path to a file with human complexity rankings
- coarse graining type (currently only 'fft' can be used)

Example:
python complexity_colour_fft.py scenes dataset_folder ranking_file.xlsx fft 

A *.csv file with resulting MSSC rankings will be stored in the results/calculated_mssc/" folder. Scatter plots and regression analysis will be stored in the results/mssc_figures/ folder.

middle_scale.py dataset_name calculates correlation with subjective rankings and plots regression graphs

Scripts scale_search.py and plot_scale_impact.py can be used for plotting the impact of scales on the overall complexity rankings and correlation of each scale ranking with human predictions for a given subset of SAVOIAS. 

Conda environment config containing all required packages is stored in mssc.yml

To reproduce our experiment and calculate MSSC and correlations for the entirety of SAVOIAS dataset (by default it's implied to be stored in the ../Savoias-Dataset folder) run:

complexity_colour_fft.sh

middle_scale.sh 

plot_scale_impact.sh 

Questions can be addressed to anna.kravchenko@ru.nl

Please reference: Kravchenko, A., Bagrov, A. A., Katsnelson, M. I., & Dudarev, V. (2024). Multi-scale structural complexity as a quantitative measure of visual complexity. arXiv preprint arXiv:2408.04076.

