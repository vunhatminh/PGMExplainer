# PGMExplainer for Graph classification

The MNIST superpixel dataset and the GNN model are obtained from https://github.com/graphdeeplearning/benchmarking-gnns

We also include their notebook *prepare_superpixels_MNIST.ipynb* in the *data/superpixels* folder.
By running it, we will generate a data file named *MNIST.pkl* in the *data/superpixels* folder.

To run PGM explainer run:
`python3 main.py --start [int1] --end [int2]`

   * int1 [optional]: start index to explain
   * int2 [optional]: end index to explain

To run GRAD benchmark run:
`python3 grad_bm.py --start [int1] --end [int2]`

   * int1 [optional]: start index to explain
   * int2 [optional]: end index to explain

To run SHAP benchmark run:
`python3 shap_bm.py --start [int1] --end [int2]`

   * int1 [optional]: start index to explain
   * int2 [optional]: end index to explain

The explanations will be saved in folder *result* in the format:
`[index, label, [node_list], [xcor_list], [ycor_list]`

   * index: index of the explained image/graph
   * label: predicted label of the explained image/graph
   * node_list: list of nodes explaining the prediction 
   * xcor_list: x coordinates of the node in node_list
   * ycor_list: y coordinates of the node in node_list

we include our notebook *visualize_explanations.ipynb* to demonstrate how to visualize the explainations.