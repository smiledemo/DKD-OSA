
## Data 
---
- [N2AwA](./data/N2AwA/classes.txt): DomainNet & AwA2
- [I2AwA](./data/I2AwA/dataset_info.txt): 3D2 & AwA2

| Dataset | Domain | Role | #Images | #Attributes | #Classes |
|:-:|:-:|:-:|:-:|:-:|:-:|
| D2AwA | A <br> P <br> R | source / target | 9,343 / 16,306 <br> 3,441 / 5,760 <br> 5,251 / 10,047 | 85 | 10 / 17 |
| I2AwA | I <br> Aw | source / target | 2,970 / 37,322 | 85 | 40 / 50 |

## Dependencies
---
- Python 3.7
- Pytorch 1.1


## Training
---
### Step 1: 
```shell
./data/N2AwA/refine_cluster-samples.ipynb
```

### Step 2: 
```shell
python main.py
```

