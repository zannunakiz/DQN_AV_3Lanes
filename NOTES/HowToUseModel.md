# How To Use Model

## Continue training from a saved model

```bash
python main_train.py --loadmodel models\model_stage1A.pth --episodes 300
```

## Visualize a saved model

```bash
python main_visualize.py --model models\model_stage1A.pth --episodes 10
```

## Visualize using tester obstacles

```bash
python main_visualize.py --model models\model_stage1A.pth --episodes 10 --tester
```

## Run across all curriculum stages

```bash
python main_visualize.py --allstage --episodes 20
```

## Alternative wrapper command

```bash
python main_visualization.py --episodes 5
```
