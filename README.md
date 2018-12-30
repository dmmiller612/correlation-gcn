# Graph Convolutional Networks on Correlation Based Graphs

This repo provides simple tools to run a Graph Convolutional Network (GCNs) on Correlation Based Graphs. 
In practice these methods can be applied to gene expression, proteomics, and other types of data. This specific 
implementation contains Graph DiffPool as well.

In correlation based graphs, the edges represent how correlated one node is with another. Due to this assumption, 
certain operations from the original GCN paper are not needed (such as summing the identity matrix with the adjacency matrix).

## How to Run the python script

I have provided a Dockerfile that can run the `example_estimator.py`\
. It takes a few arguments that can be passed in the
docker run. 


## Citing this Repo
```
@misc{Miller2018,
  author = {Miller, Derek},
  title = {Graph Convolutional Networks on Correlation Based Graphs},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dmmiller612/correlation-gcn}}
}
```

## Citations
```
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```

```
@misc{Miller2018,
  author = {Ying, Rex, You, Jiaxuan, ...},
  title = {Hierarchical Graph Representation Learning with Differentiable Pooling},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dmmiller612/correlation-gcn}}
}
```