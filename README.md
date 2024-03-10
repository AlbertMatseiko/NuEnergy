# Neutrino Selection

### Abstract

Baikal-GVD is a large (∼ 1 km3) underwater neutrino telescope located in Lake Baikal, Russia. This work presents a neural network for reconstructing energy of muons particles born in interaction of neutrino with water. Simultaneously, 
the network estimates an error of it's reconstruction in terms of 1 sigma. This achieved by using special loss function, based on gaussian likelihood.


The repository contains framework for  data analysis in [Baikal-GVD experiment](https://baikalgvd.jinr.ru/). 

Repository provides code for Physical-informed neural nets:

---

### Functionality 

- Linear and Convolution NN with fixed data
- Recurrent Neural Networks for work with sequential data

## Getting started

### Installation

Preliminaries: I recommend using Linux distributions 


Clone repository using [git](https://git-scm.com/)  
```
git clone 
```

[Poetry](https://python-poetry.org/) will install proper environment for your start 

```python

poetry init
```

### Command Line Interface

After installation project can be started solely from command: 

```
    neutrino_selection start <WRITE YOUR h5 data> --architecture <CHOOSE ARCHITECTURE>
```

Possible options are:
- NN 
- RNN
- HYPER 

### Python 

Minimal example

```python
from neutrino_selection.net import RnnHyperNetwork
from neutrino_selection.data import H5Dataset

dataset = H5Dataset('<YOUR DATA>')

net = RnnHyperNetwork(device='cuda')

net.train(dataset)
```

Proceed to tutorials for accustomation with framework

## Contact

Contact me via opening issue on github or sending email matseiko.av@phystech.edu.

Use telegram for research collaboration `@AlbertMac280`
