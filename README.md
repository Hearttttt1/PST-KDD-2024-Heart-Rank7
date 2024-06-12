# PST-KDD-2024-Heart-Rank8

## 基本环境
- Python 3.8.13
- PyTorch 2.0.1+cu117
- CUDA 11.7
- NVADIA RTX3090 24G
## 安装依赖包：
```bash
pip install -r requirements.txt
```

## 下载预训练模型并放在bert_models目录下
scibert: https://huggingface.co/allenai/scibert_scivocab_uncased/tree/main 

deberta: https://huggingface.co/microsoft/deberta-v3-base/tree/main

roberta: https://huggingface.co/allenai/dsp_roberta_base_dapt_cs_tapt_sciie_3219/tree/main

## 目录结构
```
--main
  |--data  
    |--PST
      |--paper-xml  ## 存放数据
  |--bert_models  ## 存放预训练权重
      |--deberta_v3_base
      |--dsp_roberta_base_dapt_cs_tapt_sciie_3219
      |--scibert_scivocab_uncased
  |--out  #模型输出结果保存，共六个模型
      |--kddcup
        |--deberta-base
        |--roberta-base
        |--scibert
        |--rf
  |--submit  ## 最终提交文件存放处

  |--deberta_training.ipynb  ## training后缀为训练程序
  |--deberta_test_inference.ipynb  ## inference后缀为推理程序

  |--roberta_0fold_training.ipynb
  |--roberta_0fold_test_inference.ipynb

  |--roberta_3fold_training.ipynb
  |--roberta_3fold_test_inference.ipynb

  |--scibert_0fold_training.ipynb
  |--scibert_0fold_test_inference.ipynb

  |--scibert_4fold_training.ipynb
  |--scibert_4fold_test_inference.ipynb

  |--lgb_run.ipynb  # LGB

  |--merge.ipynb # 融合模型
```
## 训练及推理过程
先运行五个模型的训练代码：
 - deberta_training.ipynb
 - roberta_0fold_training.ipynb
 - roberta_3fold_training.ipynb
 - scibert_0fold_training.ipynb
 - scibert_4fold_training.ipynb

之后运行相应的inference代码。

然后运行lgb_run.ipynb。

最后运行merge.ipynb生成最终的提交文件，保存在submit目录下。

