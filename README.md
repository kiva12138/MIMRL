# MIMRL
The implementation codes of paper: _Multimodal Sentiment Analysis with Mutual Information-based Disentangled Representation Learning. IEEE Transactions on Affective Computing. 2025._ 

In this work, we propose _Mutual Information-based Disentangled Multimodal Representation Learning_(MIMRL). Our approach involves estimating different types of information during feature extraction and fusion stages. Specifically, 
- During feature extraction: we quantitatively assess and adjust the proportions of modality-invariant,-specific, and-complementary information. 
- During fusion: we evaluate the amount of information retained by each modality in the fused representation.

We employ mutual information or conditional mutual information to estimate each type of information content.

The whole framework in shown as:
![image](https://github.com/user-attachments/assets/6c593ebb-5f69-4856-a52c-39339d4b7f4e)

As you could see, we foucus on building a whole framework for multimodal learning, meaning the interanl fusion encoders are replaceable. For convenience, we use the CubeMLP(https://github.com/kiva12138/CubeMLP) as the fusion encoder, which has been implemented in _Model.py_ and _MLPProcess.py_. If you like, you could simply replace the fusion encoders such as TFN of TensorFormer, but that may introduce some other coding works.

We use variational mutual information as a tool to estimate various types of information, but the estimation process itself also requires training. That means the training process is divided into two steps (one is for estimating mutual information, and the other is to train the model), as shown in _Model.py_. This approach introduces lots of parameters to tune (shown in _Parameters.py_, maybe more than 50), and one examples could be shown as:
```
cmd = 'python Main.py --task_name mosiDec52.1 --dataset mosi_Dec --log_scale 0-0-0 --normalize 0-1-1 --batch_size 128 --num_workers 8  '+\
'--d_common 128 --encoders gru --activate gelu --time_len 100 --d_hiddens 50-3-128=10-3-128 --d_outs 50-3-128=10-3-128 '+\
'--dropout_mlp 0.0-0.0-0.0 --dropout 0.1-0.1-0.1-0.1 --bias --res_project 1-1 '+\
'--critic_type separate --baseline_type constant --bound_type infonce --mi_lr_rate 1.0 --cmi_lr_rate 1.0 '+\
'--loss_mi_coefficient1 1-1-1-1-1-1-1-1-1-1-1 --loss_mi_coefficient2 0.01-0.01-0.01-0.01-0.01-0.01-0.01-0.01 '+\
'--k_neighbor 2 --radius 1.0 --cmi_last_acticate sigmoid --stage1_n 2 '+\
'--seed 0 --loss MAE --gradient_clip 1.5 --epochs_num 70 --optm Adam --learning_rate 4e-3 --bert_freeze no --bert_lr_rate 0.01 '+\
'--weight_decay 0.0 --lr_decrease multi_step --lr_decrease_iter 9-60 --lr_decrease_rate 0.1 --save_best_features --parallel '
print(cmd)
!{cmd}
```
For more running commands, please refer to _Run.ipynb_. We also provide several different approaches to estimating the mutual information in _VMI.py_. If you like to use them for other works, just take them and GOOD LUCK! 

Before running, be sure to change the paths in _Config.py_, in which I have also sorted some common datasets for sentiment analysis. For MOSI and MOSEI, please refer to https://github.com/kiva12138/CubeMLP to get the downloading links. For AVEC2019, I am sorry I cannot provide the processed files directly, as this is a private dataset. If you have already have licences to this dataset, please contact sunhaoxx@zju.edu.cn to get the processed files.

If you want to work with us in the multimodal field or need any help, please feel free to contact us.

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
