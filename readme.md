# CAT: Closed-loop Adversarial Training for Safe End-to-End Driving

**7th Annual Conference on Robot Learning (CoRL 2023)**


[**Webpage**](https://metadriverse.github.io/cat/) | [**Code**](https://github.com/metadriverse/cat) |[**Paper**](https://openreview.net/pdf?id=VtJqMs9ig20)




## Set Up

Clone the official implementation of CAT to local.

```bash
git clone https://github.com/metadriverse/cat.git
cd cat
```

Download (i) the modified version of MetaDrive to maneuver and display adversarial traffic in simulations  and (ii) the pre-trained DenseTNT model as the traffic prior in the safety-critical resampling. [Link](https://drive.google.com/drive/folders/1xVQ84pF5clVtKw6d4NCC-0mYbo4cIZ_a)

Place `densetnt.bin `  into the `./advgen/pretrained` folder. Your directory structure should look something like this:

```
cat
└── advgen
    └── pretrained
    	└── densetnt.bin    
└── metadrive
└── license
...
```

Finally, install dependencies via

```bash
conda create -n cat python=3.9
conda activate cat
pip install -r requirements.txt
```

## Data Preparation

We use Waymo Open Motion Dataset (WOMD) v1.1 as raw traffic scenarios and provide 500 scenarios used in our paper. [Link](https://drive.google.com/drive/folders/1xVQ84pF5clVtKw6d4NCC-0mYbo4cIZ_a)

If you want to use other cases, please follow the tutorial below.

First, download tfrecord files from the WOMD **validation/testing_interactive** folder. [Link](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario)

Second, run the script to convert them to MetaDrive scenario descriptions.

```bash
python scripts/covert_WOMD_to_MD.py
```

Third, select scenarios that lasts 9.1 seconds and contains 2 vehicles labeled as Objects of Interest (one is the ego vehicle, the other is designated as the opponent vehicle). Currently, CAT supports 1 ego + 1 opponent + n other vehicles in one scenario.

```
python scripts/select_cases.py
```

## Visualize the safety-critical scenario generation

Run the following script to visualize how CAT dynamically generates safety-critical scenarios and benchmark the attack success rate and computational time. 

```bash
python cat_advgen.py
```

The safety-critical scenario generation pipeline is universal with respect to arbitrary ego controllers. In this example, we generate adversarial traffic against EgoReplay policy. You can replace it with your own policies.

## Train a TD3-based policy with CAT  

Run the following script to conduct CAT training.

```bash
python cat_RLtrain.py --mode cat --seed 0
```

Run the following script to visualize the learning curves about route completions and crash rates.

```bash
./scripts/plot.sh
```

Log files for testing the refactored codebase are placed in the `./testlogs` folder. 

## Reference

```latex
@inproceedings{zhang2023cat,
  title={CAT: Closed-loop Adversarial Training for Safe End-to-End Driving},
  author={Zhang, Linrui and Peng, Zhenghao and Li, Quanyi and Zhou, Bolei},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```

The traffic prior model is heavily based on DenseTNT. If you find the code useful for your research, please kindly consider citing their paper:

```latex
@inproceedings{densetnt,
  title={Densetnt: End-to-end trajectory prediction from dense goal sets},
  author={Gu, Junru and Sun, Chen and Zhao, Hang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15303--15312},
  year={2021}
}
```

