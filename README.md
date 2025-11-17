# IIVFormer: Intra and Inter View Transformer for 3D Human Pose Estimation

## Environment
The project is developed under the following environment:

python 3.13.5 

PyTorch 2.8.0

CUDA 12.9


For installation of the project dependencies, please run:

```
pip install -r requirements.txt
```


## Dataset

The Human3.6M dataset and HumanEva dataset setting follow the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). Please refer to it to set up the Human3.6M dataset (under ./data directory).

The MPI-INF-3DHP dataset setting follows the [P-STMO](https://github.com/paTRICK-swk/P-STMO). Please refer it to set up the MPI-INF-3DHP dataset (also under ./data directory).


## Training from scratch

To train our model using the CPN's 2D keypoints as inputs please run:
```
python main.py ----data_2d data/data_2d_h36m_cpn_ft_h36m_dbb.npz --lr 0.0004
```

## Evaluating

To evaluate our model using the CPN's 2D keypoints as inputs, please run:

```
python evaluate.py --model_path 'your model path'
```




## Visualization
Please refer to the [MHFormer](https://github.com/Vegetebird/MHFormer).
