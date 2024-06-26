# spatiotemporal-network
## Abstract: 
The present study aims to infer individuals' social networks from their spatiotemporal behavior acquired via wearable sensors. Previously proposed static network metrics (e.g., centrality measures) cannot capture the complex temporal patterns in dynamic settings (e.g., children's play in a schoolyard). Moreover, the existing temporal metrics often overlook the spatial context of interactions. This study aims firstly to introduce a novel metric on social networks in which both temporal and spatial aspects of the network are considered to unravel the spatiotemporal dynamics of human behavior. This metric can be used to understand how individuals utilize space to access their network and how individuals are accessible by their network. We evaluate the proposed method on real data to show how the proposed metric enhances the performance of a clustering task. Secondly, this metric is used to interpret interactions in a real-world dataset collected from children playing in a playground. Our experiments show performance improvements over the existing temporal measures. Furthermore, by considering spatial features, this metric provides unique knowledge of the spatiotemporal accessibility of individuals in a community and more clearly captures pairwise accessibility compared with existing temporal metrics. Thus, it can facilitate domain scientists interested in understanding social behavior in the spatiotemporal context. We further make our collected dataset publicly available, allowing further research.

<img src="[https://github.com/favicon.ico](https://github.com/maaeedee/spatiotemporal-network/assets/20282362/e744f10b-908e-498d-9a45-883e914593d4)" width="48">

## Link to the paper: 
[To be added soon.]
## Required libraries: 
[To be added soon.]

## How to run the code:
In order to run the code for this analysis, you need to obtain spatio temporal data. In this repository, the dummy examples are provided (see Data/Dummy). You could also download the Spatiotemporal data provided in this paper via this link. After preparing data, you need to simply run the main python file as follows:


```python spatiotemporal_dummy_main.py```

## How to site this work:

```
@article{nasri2023novel,
  title={A novel metric to measure spatio-temporal proximity: a case study analyzing children’s social network in schoolyards},
  author={Nasri, Maedeh and Baratchi, Mitra and Tsou, Yung-Ting and Giest, Sarah and Koutamanis, Alexander and Rieffe, Carolien},
  journal={Applied Network Science},
  volume={8},
  number={1},
  pages={50},
  year={2023},
  publisher={Springer}
}

```

