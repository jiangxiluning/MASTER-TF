# MASTER-TensorFlow ![](https://img.shields.io/badge/license-MIT-blue)

<div align=center>
<img src="https://github.com/wenwenyu/MASTER-pytorch/blob/main/assets/logo.jpeg" width="200" height="200" />
</div>


TensorFlow reimplementation of ["MASTER: Multi-Aspect Non-local Network for Scene Text Recognition"](https://arxiv.org/abs/1910.02562)
(Pattern Recognition 2021). This project is different from our original implementation that builds on the privacy codebase FastOCR of the company.
You can also find PyTorch reimplementation at [MASTER-pytorch](https://github.com/wenwenyu/MASTER-pytorch) repository,
and the performance is almost identical. (PS. Logo inspired by the Master Oogway in Kung Fu Panda)


## News
* 2021/07: [MASTER-mmocr](https://github.com/JiaquanYe/MASTER-mmocr), reimplementation of MASTER by mmocr. [@Jianquan Ye](https://github.com/JiaquanYe)
* 2021/07: [TableMASTER-mmocr](https://github.com/JiaquanYe/TableMASTER-mmocr), 2nd solution of ICDAR 2021 Competition on Scientific Literature Parsing Task B based on MASTER. [@Jianquan Ye](https://github.com/JiaquanYe)
* 2021/05: [Savior](https://github.com/novioleo/Savior), which aims to provide a simple, lightweight, fast integrated, pipelined deployment framework for RPA,
  is now integrated MASTER for captcha recognition. [@Tao Luo](https://github.com/novioleo)
* 2021/04: Slides can be found at [here](https://github.com/wenwenyu/MASTER-pytorch/blob/main/assets/MASTER.pdf).


## Honors based on MASTER
* 1st place (2021/05) solution to [ICDAR 2021 Competition on Scientific Table Image Recognition to LaTeX (Subtask I: Table structure reconstruction)](https://competitions.codalab.org/competitions/26979)
* 1st place (2021/05) solution to [ICDAR 2021 Competition on Scientific Table Image Recognition to LaTeX (Subtask II: Table content reconstruction)](https://competitions.codalab.org/competitions/26979)
* 2nd place (2021/05) solution to [ICDAR 2021 Competition on Scientific Literature Parsing Task B: Table recognition](https://icdar2021.org/program-2/competitions/competition-on-scientific-literature-parsing/)
* 1st place (2020/10) solution to [ICDAR 2019 Robust Reading Challenge on Reading Chinese Text on Signboard (task2)](https://rrc.cvc.uab.es/?ch=12&com=evaluation&task=2)
* 2nd and 5th places (2020/10) in [The 5th China Innovation Challenge on Handwritten Mathematical Expression Recognition](https://www.heywhale.com/home/competition/5f703ac023f41e002c3ed5e4/content/6)
* 4th place (2019/08) of [ICDAR 2017 Robust Reading Challenge on COCO-Text (task2)](https://rrc.cvc.uab.es/?ch=5&com=evaluation&task=2)
* More will be released


## Introduction
MASTER is a self-attention based scene text recognizer that (1) not only encodes the input-output attention,
but also learns self-attention which encodes feature-feature and target-target relationships inside the encoder
and decoder and (2) learns a more powerful and robust intermediate representation to spatial distortion and
(3) owns a better training and evaluation efficiency. Overall architecture shown follows.

<div align=center>
<img src="https://github.com/wenwenyu/MASTER-pytorch/blob/main/assets/overall.png" />
</div>
          
This repo contains the following features.

- [x] Multi-gpu Training
- [x] Greedy Decoding
- [x] Single image inference
- [x] Eval iiit5k
- [x] Convert Checkpoint to SavedModel format
- [x] Refactory codes to be more tensorflow-style and be more consistent to graph mode
- [x] Support tensorflow serving mode


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

**Since I made change to the usage of gcb block, the weight could not be suitable to HEAD. If you want to test the model, please use https://github.com/jiangxiluning/MASTER-TF/commit/85f9217af8697e41aefe5121e580efa0d6d04d92**

Currently, you can download checkpoint from [here](https://pan.baidu.com/s/1ijpo8WRZHR-AyDclxQVDiw) with code **o6g9**, or from [Google Driver](https://drive.google.com/file/d/1gpfMvnQWZimogQLFM_teOwiLNz-ZEF02/view?usp=sharing), this checkpoint was trained with MJ and selected
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

## Tensorflow Serving

For tensorflow serving, you should use savedModel format, I provided test case to show you how to convert a checkpoint to savedModel and how to use it.

```bash
pytest -s tests/test_units::test_savedModel  #check the test case test_savedModel in tests/test_units
pytest -s tests/test_units::test_loadModel  # call decode to inference and get predicted transcript and logits out.
```


## Citations
If you find MASTER useful please cite our [paper](https://arxiv.org/abs/1910.02562):
```bibtex
@article{Lu2021MASTER,
  title={{MASTER}: Multi-Aspect Non-local Network for Scene Text Recognition},
  author={Ning Lu and Wenwen Yu and Xianbiao Qi and Yihao Chen and Ping Gong and Rong Xiao and Xiang Bai},
  journal={Pattern Recognition},
  year={2021}
}
```


## License
This project is licensed under the MIT License. See LICENSE for more details.

## Acknowledgements

Thanks to the authors and their repo:
 - [SAR_TF](https://github.com/Pay20Y/SAR_TF)
 - [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
