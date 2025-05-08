# MIMRL
The implementation codes of paper: _Multimodal Sentiment Analysis with Mutual Information-based Disentangled Representation Learning. IEEE Transactions on Affective Computing. 2025._ 

In this work, we propose _Mutual Information-based Disentangled Multimodal Representation Learning_(MIMRL). Our approach involves estimating different types of information during feature extraction and fusion stages. Specifically, 
- During feature extraction: we quantitatively assess and adjust the proportions of modality-invariant,-specific, and-complementary information. 
- During fusion: we evaluate the amount of information retained by each modality in the fused representation.

We employ mutual information or conditional mutual information to estimate each type of information content.

The whole framework in shown as:


As you could see, we foucus on building a whole framework for multimodal learning, meaning the interanl fusion encoders are replaceable. For convenience, we use the CubeMLP as the fusion encoder, which has been implemented in _Model.py_ and _MLPProcess.py_. 


If you find it's useful for you, please consider citing our work:
```
@article{sun2025multimodal,
  title={Multimodal Sentiment Analysis with Mutual Information-based Disentangled Representation Learning},
  author={Sun, Hao and Niu, Ziwei and Wang, Hongyi and Yu, Xinyao and Liu, Jiaqing and Chen, Yen-Wei and Lin, Lanfen},
  journal={IEEE Transactions on Affective Computing},
  year={2025},
  publisher={IEEE}
}
```
