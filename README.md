# smile-network-localization

This repository contains code supporting "SMILE: Robust Network Localization via Sparse and Low-Rank Matrix Decomposition". For any questions, please reach out to Lilly Clark (lilliamc@usc.edu) or Sampad Mohanty (sbmohant@usc.edu).

To test SMILE, our novel approach to network localization, on a simulated dataset of 500 nodes with 50 anchors, run `smile.py`.

To test generating simulated data, run `process_data.py`.

To test a graph convoluntional network for large-scale network localization, run `gcn.py`.
For more details, see https://github.com/Yanzongzi/GNN-For-localization.

To test a baseline approach of rank reduction via PSVD and embedding via multidimensional scaling, run `mds.py`.

To compare each of these approaches on a network of 500 nodes and 50 anchors, run `run_test.py`.

**Requirements**  
python 3.6.9  
torch 1.10.2  
torch-geometric 2.0.3  
numpy 1.18.5
