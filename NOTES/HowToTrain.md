# How To Train

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Start a normal training run

```bash
python main_train.py --episodes 500
```

## 3. Optional useful variants

Train with live rendering:

```bash
python main_train.py --episodes 200 --visualize --render-every 5
```

Train faster with batched environment steps:

```bash
python main_train.py --episodes 500 --multiply --multivalid
```

Train with fixed seed:

```bash
python main_train.py --episodes 500 --seed 123
```

## 4. Output location

New checkpoints and logs are written to `models/` by the training flow.
