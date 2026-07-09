# Random-Sampling-SNN

### Requirements

pytorch==1.10.0+cu113

spikingjelly==0.0.0.0.12

networkx==3.1

timm==1.0.9

torchtoolbox==0.1.8.2


### Training on ImageNet

```
python -m torch.distributed.launch --nproc_per_node=8 main.py --dataset IMAGENET --batch-size 16 --epochs 120 --T 4 --graph-model WS --skip-ratio 0.05 --data-path YOUR_DATA_PATH
```



### Training on CIFAR-10

````
python -m torch.distributed.launch --nproc_per_node=1 main.py --dataset CIFAR-10 --graph-model BA --skip-ratio 0.05 --data-path YOUR_DATA_PATH
````



### Training on CIFAR-100

````
python -m torch.distributed.launch --nproc_per_node=1 main.py --dataset CIFAR-100 --graph-model BA --skip-ratio 0.05 --data-path YOUR_DATA_PATH
````

### Problems you may meet
If you get an output like follows:
AttributeError: module 'numpy' has no attribute 'int'.
`np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.

You can fix it by replace `np.int` by `int`.
### What i have done 
I first studied SNN and its working ,like why it came into picture.
And then we directly dived into the connection of SCN modules .The arrangement of these modules were done by some different topologies as the already exsisting ANN architectures were not able to unleash the true potential of SNN.We then studied the graphs used in the research paper namely WS ,ER,GNM,BA.The highest accuracy was given by WS so our sole aim of this research was to beat this WS model by using some new type of arrangement which is small world .So then we started testing some new models .Initially were trying random rewiring method like WS but the results were not as per our expectation,so we moved to the graph arrangement which was solely based on some mathematical formulas ,The most successful of them were the 
1)based on gravity equation.
2) based on cosine distribution
and research is still going on.
3)Based on mimizing the Hamiltonian energy

