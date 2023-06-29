# Saar-Final-Project

Saar Poplinger's EE B.Sc. final project code files.

## General run notes:

• .lsf files - Lumerical workspace files - Open the code file in the corresponding simulation file (from Google Drive) and run (or change parameters first and then run). Link to Google Drive: https://drive.google.com/drive/folders/1PX2SdqVLsQWwZVu5lMxyiQL04COi2s5a?usp=share_link

• .ipynb file - Jupyter notebook (python) files, they run in blocks. Just run each block according to instructions in the code.

• .py files - the notebooks but in a regular file. For the workflow to be the same as the project's, it has to run in blocks the same way as the .ipynb files.

## The files in the repository:

• simulation_sweeper_analyzer_te.lsf, simulation_sweeper_analyzer.lsf - create Lumerical simulations for dataset for TE or TM, accordingly.

• MLOptTE2Adj.py/.ipynb, MLOptTMAdj.py/.ipynb - create nn and perform optimization, return optimal geometry for TE or TM, accordingly.

• simulation_spectrum_analyze_te.lsf, simulation_spectrum_analyze.lsf - create final check Lumerical simulations for TE or TM, accordingly
