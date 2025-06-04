# 🚀 Progressive Weight Loading (PWL)

**Progressive Weight Loading (PWL)** is a novel technique designed to balance fast model initialization with high performance in deep learning inference.

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-blue" />
  <img src="https://img.shields.io/badge/Distillation-Knowledge-green" />
  <img src="https://img.shields.io/badge/Edge%20AI-Ready-orange" />
</p>

---

## 🧠 What is PWL?

PWL starts by loading a compact **student model**, allowing for **fast initial inference**, and then **progressively replaces** its layers with those of a larger **teacher model**. This enables:

- ⚡ **Low-latency inference at startup**  
- 🔁 **Gradual performance improvement**
- 📱 **Better deployment flexibility for mobile and resource-constrained environments**

---

## 📦 What’s in This Repository?

This repository includes:

- 🔧 Code for training student models with PWL-based knowledge distillation  
- 🧪 Scripts for evaluation and progressive weight substitution  
- 📊 Example configurations using **VGG**, **ResNet**, and **ViT**

---


## 🏁 How to Start

### 📥 Installation

```bash
git clone https://github.com/5yearsKim/ProgressiveWeightLoading
cd ProgressiveWeightLoading
mkdir ckpts
```

### ⚙️ Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

---

## 💾 Save Model Config

Before training, save the model configuration for both student and teacher using:

```bash
python scripts/save_model_config.py \
  -m {resnet-teacher,resnet-student,lenet5-teacher,lenet5-student,vgg-teacher,vgg-student,vit-teacher,vit-student} \
  -d {cifar100,cifar10,imagenet}
```

---

## 🎓 Train the Teacher Model

Train a teacher model from scratch:

```bash
python train_teacher.py \
  --model_type {lenet5,resnet,vgg,vit} \
  --data_type {cifar10,cifar100} \
  --epochs 100 --lr 0.01 --bs 64 \
  --pretrained_path ckpts/teacher --save_path ckpts/teacher \
  --experiment_name PWL_Teacher
```

For fine-tuning (e.g., ViT):

```bash
python finetune_teacher.py \
  --model_type vit --data_type cifar100 \
  --epochs 50 --lr 0.001 --bs 64 \
  --teacher_config_path configs/vit_teacher.yaml \
  --pretrained_path ckpts/pretrained_vit \
  --save_path ckpts/teacher \
  --experiment_name PWL_Teacher_FT
```

---

## 🔁 Distill the Student Model

Prepare a trained teacher and distill the student model:

```bash
python train_student.py \
  --model_type {resnet,lenet5,vgg,vit} \
  --data_type {cifar10,cifar100,imagenet} \
  --teacher_path ckpts/teacher \
  --student_path configs/student.yaml \
  --student_pretrained_path ckpts/student_pretrained \
  --output_dir ckpts/student_distilled \
  --epochs 100 --lr 0.01 --min_lr 1e-5 --bs 64 \
  --experiment_name PWL_Student \
  --cross_mode {random,all}
```

---

## 📊 Evaluate Results

Evaluate the distilled or progressively-loaded model:

```bash
python eval_swapnet.py \
  --model_type {resnet,lenet5,vgg,vit} \
  --data_type {cifar10,cifar100} \
  --model_path ckpts/student_distilled \
  --batch_size 64
```

---

## 📝 Notes
- `--is_sample` can be used for debugging or testing with a smaller dataset.
- All scripts are integrated with **MLflow** for experiment tracking.
- Models and configs are stored in `ckpts/` directory.



### 📊 Accuracy under Various Teacher‐Layer Loading Orders

Accuracy results on CIFAR-10 and CIFAR-100 datasets for VGG, ResNet, and ViT using different teacher-layer loading strategies.

#### 🔹 Prefix Loading

| Loading Order                           | VGG (C10) | ResNet (C10) | ViT (C10) | VGG (C100) | ResNet (C100) | ViT (C100) |
|----------------------------------------|-----------|--------------|-----------|-------------|----------------|-------------|
| Student (`S₁S₂S₃S₄`)                   | 91.7      | 92.3         | 94.3      | 71.1        | 72.1           | 74.6        |
| `T₁→S₂→S₃→S₄`                          | 92.1      | 91.9         | 95.5      | 70.6        | 71.4           | 76.2        |
| `T₁→T₂→S₃→S₄`                          | 93.1      | 92.9         | 96.4      | 72.5        | 72.8           | 79.8        |
| `T₁→T₂→T₃→S₄`                          | 92.7      | 94.4         | 97.1      | 73.8        | 73.6           | 81.4        |
| Teacher (`T₁T₂T₃T₄`)                   | 93.8      | 94.8         | 97.4      | 74.2        | 75.7           | 82.3        |

#### 🔹 Suffix Loading

| Loading Order                           | VGG (C10) | ResNet (C10) | ViT (C10) | VGG (C100) | ResNet (C100) | ViT (C100) |
|----------------------------------------|-----------|--------------|-----------|-------------|----------------|-------------|
| Student (`S₁S₂S₃S₄`)                   | 91.7      | 92.3         | 94.3      | 71.1        | 72.1           | 74.6        |
| `S₁→S₂→S₃→T₄`                          | 91.0      | 88.8         | 82.1      | 65.6        | 69.2           | 61.3        |
| `S₁→S₂→T₃→T₄`                          | 90.3      | 93.4         | 84.2      | 67.6        | 73.7           | 64.7        |
| `S₁→T₂→T₃→T₄`                          | 91.4      | 94.3         | 87.5      | 69.1        | 74.5           | 68.8        |
| Teacher (`T₁T₂T₃T₄`)                   | 93.8      | 94.8         | 97.4      | 74.2        | 75.7           | 82.3        |

#### 🔹 Contiguous Block Loading

| Loading Order                           | VGG (C10) | ResNet (C10) | ViT (C10) | VGG (C100) | ResNet (C100) | ViT (C100) |
|----------------------------------------|-----------|--------------|-----------|-------------|----------------|-------------|
| Student (`S₁S₂S₃S₄`)                   | 91.7      | 91.8         | 94.3      | 53.1        | 72.1           | 74.6        |
| `S₁→T₂→S₃→S₄`                          | 86.1      | 88.1         | 84.5      | 59.4        | 65.2           | 62.1        |
| `S₁→S₂→T₃→S₄`                          | 88.4      | 90.2         | 81.2      | 64.1        | 68.8           | 59.7        |
| `S₁→T₂→T₃→S₄`                          | 89.8      | 89.1         | 78.2      | 65.8        | 71.0           | 58.3        |
| Teacher (`T₁T₂T₃T₄`)                   | 93.8      | 94.2         | 97.4      | 56.1        | 75.7           | 82.3        |

