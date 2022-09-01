# Patefon: Transformer for Topological Data

This is unofficial implementation of Persformer architecture, described in [Reinauer et al](https://arxiv.org/abs/2112.15210). 

Patefon is Transformer-like model that takes homologies as input. It can be used for classification and regression of topological data.

<img src="https://i.postimg.cc/nzqjCy3b/kisspng-patefon-gramophone-phonograph-clip-art-phonograph-5b48edc70fae87-4472405115315061190642.png"  width="150">

The architecture is the same as in the orginial paper about Persformer except for the way homologies are embedded. While in Persformer all homologies are represented by their birth and death times, in Patefon 0th homologies are represented by only their death times. Thus, 0th and 1st homologies pass through different linear layers.

To test Patefon on ORBIT5K dataset, run

```
git clone https://github.com/Markfryazino/patefon
cd patefon
pip install -r requirements.txt
python3 run_training.py
```

`Ripser-plusplus` is used for computation of homologies.