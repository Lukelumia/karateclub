 ![GitHub stars](https://img.shields.io/github/stars/benedekrozemberczki/karateclub.svg?style=plastic) ![GitHub forks](https://img.shields.io/github/forks/benedekrozemberczki/karateclub.svg?color=blue&style=plastic) ![License](https://img.shields.io/github/license/benedekrozemberczki/karateclub.svg?color=blue&style=plastic)

<p align="center">
  <img width="90%" src="https://github.com/benedekrozemberczki/karateclub/blob/master/karatelogo.jpg?sanitize=true" />
</p>

--------------------------------------------------------------------------------


**[Documentation](https://karateclub.readthedocs.io/)** | **[Paper](https://arxiv.org/abs/1802.03997)** 

*karateclub* is a community detection extension library for [NetworkX](https://networkx.github.io/).


Karate Club consists of state-of-the-art methods to do overlapping and non-overlapping clustering on graphs. The package also includes methods that can deal with bipartite, temporal and heterogeneous graphs. Implemented methods cover a wide range of network science (NetSci, Complenet), data mining (ICDM, CIKM, KDD), artificial intelligence (AAAI, IJCAI) and machine learning (NeurIPS, ICML, ICLR) conferences.  

--------------------------------------------------------------------------------

Karate Club makes the use of modern community detection tecniques quite easy (see [here](link_here) for the accompanying tutorial).
For example, this is all it takes to use on a Watts-Strogatz graph [Ego-splitting](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf):

```python
import networkx as nx
from karateclub import EgoNetSplitter

g = nx.newman_watts_strogatz_graph(1000, 20, 0.05)

splitter = EgoNetSplitter(1.0)

splitter.fit(g)

print(splitter.overlapping_partitions)
```

In detail, the following methods are currently implemented:

* **[Ego-Splitting](https://karateclub.readthedocs.io/en/latest/modules/root.html#karateclub.ego_splitter.EgoNetSplitter)** from Epasto *et al.*: [Ego-splitting Framework: from Non-Overlapping to Overlapping Clusters](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf) (KDD 2017)

* **[EdMot](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf)** from Li *et al.*: [EdMot: An Edge Enhancement Approach for Motif-aware Community Detection](https://arxiv.org/abs/1906.04560) (KDD 2019)

--------------------------------------------------------------------------------

Head over to our [documentation](https://karateclub.readthedocs.io) to find out more about installation and data handling, a full list of implemented methods, and datasets.
For a quick start, check out our [examples](https://github.com/benedekrozemberczki/karateclub/tree/master/examples.py).

If you notice anything unexpected, please open an [issue](https://github.com/benedekrozemberczki/karateclub/issues) and let us know.
If you are missing a specific method, feel free to open a [feature request](https://github.com/benedekrozemberczki/karateclub/issues).
We are motivated to constantly make Karate Club even better.

## Installation

```sh
$ pip install karateclub
```


## Running examples

```
$ python examples.py
```
