# SiamGLIP
models: [百度云](https://pan.baidu.com/s/1lcOpUl9A8IEQ-fKHyTUpoQ?pwd=8827) (提取码: 8827)

raw_results: [百度云](https://pan.baidu.com/s/17a0bKuEMgNVlmtIRXRxJng?pwd=8827) (提取码: 8827)


![Uploading framework.png…]()


## Install the environment
Our implementation is based on PyTorch 3.9.23+CUDA 11.8. Use the following command to install the runtime environment:

```
conda env create -f SiamGLIP_env_cuda118.yaml
conda activate siamglip
```
## Data Preparation
```
${PROJECT_ROOT}
 -- data
     -- antiuav410
         |-- train
             |-- 01_1667_0001-1500
             ...
             |-- list.txt
             |-- train.json
         |-- test
             |-- 02_6319_1500-2999
             ...
         |-- val
             |-- 03_7951_1500-2999
             ...
             |-- list.txt
             |-- val.json
     -- antiuav310
         |-- test
         |-- train
         |-- val
```
## Training
Download pre-trained model [百度云](https://pan.baidu.com/s/1lcOpUl9A8IEQ-fKHyTUpoQ?pwd=8827) (提取码: 8827) and place it under PROJECT_ROOT/libs/mmdetection-main/checkpoints/.
```
torchrun --nproc_per_node=4 tracking_train_demo.py --launcher pytorch
```
## Evaluation
Download the model weights and put the downloaded weights on PROJECT_ROOT/work_dirs.

```
python tracking_test_demo.py
```





















