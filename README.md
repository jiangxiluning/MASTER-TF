# This is a reimplementation of [MASTER](https://arxiv.org/abs/1910.02562).

MASTER is a scene text recognition model which is based on self-attention mechanism. Below is the architecture.
![WX20200703-001140.png](https://i.loli.net/2020/07/03/Nj1CPvrT7J2ehWy.png)

This repo is a tensorflow implemention which may differ from our pytorch implementaion when we have done in PingAn for paper.
            
            
This repo is its tensorflow implemention.

- [x] Multi-gpu Training
- [x] Greedy Decoding
- [x] Single image inference
- [x] Eval iiit5k


## Preparation  
It is highly recommended that install tensorflow-gpu using conda.

Python3.7 is preferred.

```bash
pip install -r requirements.txt
```

## Dataset


I use Clovaai's MJ training split for training. 

please check `src/dataset/benchmark_data_generator.py` for details.

Eval datasets are some real scene text datasets. You can downloaded directly from [here](https://drive.google.com/drive/folders/1OG4ufr-kj2jFLmM4gyFEI0tMGYZrz8HI).

## Training

```bash
# training from scratch
python train.py -c [your_config].yaml

# resume training from last checkpoint
python train.py -c [your_config].yaml -r

# finetune with some checkpoint
python train.py -c [your_config].yaml -f [checkpoint]
```


## Eval

Currently, you can download checkpoint from [here](https://pan.baidu.com/s/1ijpo8WRZHR-AyDclxQVDiw) with code **o6g9**, this checkpoint was trained with MJ and selected
for the best performance of iiit5k dataset. Below is the comparision between pytorch version and tensorflow version.

| Framework | Dataset | Word Accuracy | Training Details |
| --- | --- | --- | --- |
| Pytorch | MJ | 85.05% | 3 V100 4 epochs Batch Size: 3*128|
| Tensorflow | MJ | 85.53% | 2 2080ti 4 epochs Batch Size: 2 * 50 |



Please download the checkpoint and model config from [here](https://pan.baidu.com/s/1ijpo8WRZHR-AyDclxQVDiw) with code **o6g9** and unzip it, and you can get this metric by running:

```bash
python eval_iiit5k.py --ckpt [checkpoint file] --cfg [model config] -o [output dir] -i [iiit5k lmdb test dataset]
```
The checkpoint file argument should be `${where you unzip}/backup/512_8_3_3_2048_2048_0.2_0_Adam_mj_my/checkpoints/OCRTransformer-Best` 

## Citations
If you find this code useful please cite our [paper](https://arxiv.org/abs/1910.02562):
```bibtex
@misc{lu2019master,
    title={MASTER: Multi-Aspect Non-local Network for Scene Text Recognition},
    author={Ning Lu and Wenwen Yu and Xianbiao Qi and Yihao Chen and Ping Gong and Rong Xiao},
    year={2019},
    eprint={1910.02562},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License
This project is licensed under the MIT License. See LICENSE for more details.

## Acknowledgements

Thanks to the authors and their repo:
 - [SAR_TF](https://github.com/Pay20Y/SAR_TF)
 - [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
