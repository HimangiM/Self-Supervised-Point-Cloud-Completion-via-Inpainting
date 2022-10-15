# Self-Supervised-Point-Cloud-Completion-via-Inpainting, BMVC 2021 (Oral).

Authors: Himangi Mittal, Brian Okorn, Arpit Jangid, David Held

[[Conference Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0443.pdf)][[Arxiv (Paper + Supplementary)](https://arxiv.org/abs/2111.10701)][[Conference Presentation](https://www.bmvc2021-virtualconference.com/conference/papers/paper_0443.html)][[Webpage](https://self-supervised-completion-inpainting.github.io/)]

### Citation
If you find our work useful in your research, please cite:
```
@article{mittal2021self,
  title={Self-Supervised Point Cloud Completion via Inpainting},
  author={Mittal, Himangi and Okorn, Brian and Jangid, Arpit and Held, David},
  journal={arXiv preprint arXiv:2111.10701},
  year={2021}
}
```

### Introduction
In this work, we adopt an inpainting-based approach for self-supervised point cloud completion to train our network using only partial point clouds. Given a partial point cloud as input, we randomly remove regions from it and train the network to complete these regions using the input as the pseudo-ground truth. The loss is only applied to the regions which have points in the observed input partial point cloud. Since, the network cannot differentiate between synthetic and natural occlusions, the network predicts a complete point cloud. 

For more details, please refer to our [paper](https://arxiv.org/abs/2111.10701) or [project page](https://self-supervised-completion-inpainting.github.io/).

### Installation 
(a). Clone the repository
```
git clone https://github.com/HimangiM/Self-Supervised-Point-Cloud-Completion-via-Inpainting.git
```

b). Install dependencies
```
python3 -m venv pcn3
source pcn3/bin/activate
cd Self-Supervised-Point-Cloud-Completion-via-Inpainting
pip install -r requirements.txt
Build point cloud distance ops by running `make` under `pc_distance`.
```

### Training
To train on Shapenet categories (airplane, car, chair respectively), run the following command:
```
Airplanes: bash command_train_shapenet_planes.sh
Cars: bash command_train_shapenet_cars.sh
Chairs: bash command_train_shapenet_chairs.sh
```

To train on Semantic KITTI dataset, run the following command:

```
bash command_train_semantickitti.sh
```

### Testing
Simply, change the --train flag to --test flag in the command file and call lmdb_train dataset.
