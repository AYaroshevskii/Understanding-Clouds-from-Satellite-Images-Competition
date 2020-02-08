# Understanding-Clouds-from-Satellite-Images-Competition
Simple solution (41th private leaderboard) of Understanding [Clouds from Satellite Images competition](https://www.kaggle.com/c/understanding_cloud_organization)

#### Model Description
Final solution is based on ensembling of two types of models - FPN and Unet with different backbones (efficientnet and densenet)<br/>
See more information about [pytorch segmentation models](https://github.com/qubvel/segmentation_models.pytorch)

#### Installation
```
$ pip3 install -r requirements.txt
```

#### Prepare dataset
Download and extract data from competition to main directory.

#### Train Model

```
$ python3 train.py --encoder=b2 --fold=0 --batch_size=8 
                   --model=Unet --seed=123 --max_epoch=40 
                   --device=cpu --start_epoch=0
```

#### Test Inference
```
$ python3 test_inference.py --encoder=b2 --model=Unet --batch_size=8 --model_checkpoint=b2_fold0.pth
or
$ python3 test_inference.py --encoder=b2 --model=Unet --fold=0 --batch_size=8
```

#### Further Improvements
To get higher score try to train model with BCEDICE Loss, add to segmentation model another classification head.
