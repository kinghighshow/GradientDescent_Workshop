# Gradient Descent Workshop — Group 1

**Course:** CSCN8010 – Foundations of Machine Learning Frameworks (Conestoga College)
**Instructor:** Prof. David Espinosa
**Repository:** https://github.com/kinghighshow/GradientDescent_Workshop

This workshop studies gradient descent, the optimization method used to train neural networks: the mathematics behind it, and the same training logic implemented in **PyTorch**, **TensorFlow** (`tf.GradientTape`) and **Keras** (`model.fit()` and a custom training loop), followed by a reflection comparing the three frameworks.

## Team

| Member | Student ID | Part |
|---|---|---|
| Sultan Akinola Atanda | 9114837 | Mathematical understanding, merged notebook, reflections Q1–Q5, Q14–Q15 |
| Antonio Sainz | 9072844 | PyTorch implementation (Challenge A), README, reflections Q6, Q9, Q10 |
| John Buni | 9115726 | TensorFlow implementation (Challenge B + extension), submission PDF, reflections Q7, Q8, Q11 |
| Echezona (Eche) Oji | 9078881 | Keras implementation (Challenges C1, C2), requirements and reproducibility, reflections Q12, Q13 |

---

## 1. Getting started

### Prerequisites

- Python 3.12 (64-bit)
- Git
- VS Code with the Python and Jupyter extensions, or JupyterLab

### Clone the repository

```bash
git clone https://github.com/kinghighshow/GradientDescent_Workshop.git
cd GradientDescent_Workshop
```

### Create the environment and install the dependencies

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` installs PyTorch, TensorFlow, Plotly, NumPy, Matplotlib and the Jupyter kernel. On Windows and macOS, pip installs the CPU build of PyTorch, which is enough for this workshop.

### Check the installation

```bash
python -c "import torch, tensorflow as tf, plotly; print(torch.__version__, tf.__version__, plotly.__version__)"
```

Run it with the environment's Python (`.\.venv\Scripts\python.exe` on Windows). It should print the three versions without errors.

### Run the notebooks

1. Open the folder in VS Code.
2. Open a notebook and select the kernel **`.venv`** (top-right corner, *Select Kernel → Python Environments → .venv*).
3. Click **Run All**.

To run a notebook from the terminal instead (requires `pip install nbconvert`):

```bash
jupyter nbconvert --to notebook --execute --inplace individual_notebooks/GD_Antonio.ipynb
```

Every notebook should finish without errors and show its Plotly charts. Results are reproducible because each experiment fixes its random seed.

### Repository structure

```
GradientDescent_Workshop/
├── GradientDescent_Workshop.ipynb   # Professor's original workshop notebook (unchanged)
├── individual_notebooks/
│   ├── GD_Antonio.ipynb             # Challenge A – PyTorch
│   ├── GD_John.ipynb                # Challenge B – TensorFlow
│   ├── GD_Eche.ipynb                # Challenges C1 and C2 – Keras
│   └── GD_sultan_.ipynb             # Mathematical understanding
├── images/                          # Static copies of charts shown in this README
├── requirements.txt
└── README.md
```

---

## 2. Workshop content and results

### Mathematical foundations (Sultan)

*[Placeholder — summary of the explanations of loss, gradient, learning rate, parameter update and backpropagation, to be added after the merge.]*

### Challenge A — PyTorch (Antonio)

Notebook: [`individual_notebooks/GD_Antonio.ipynb`](individual_notebooks/GD_Antonio.ipynb)

Model $\hat{y} = xw$ trained on $x = [1,2,3,4]$, $y = [2,4,6,8]$ (true weight $w = 2$) with mean squared error, starting from the same random weight (1.5410, seed 0) in every experiment.

| Experiment | Setting | Learned weight | Final MSE loss |
|---|---|---|---|
| Manual gradient descent with autograd | lr = 0.01, 200 epochs | 2.000000 | 9.0 × 10⁻¹³ |
| Same loop without `w.grad.zero_()` | lr = 0.01, 200 epochs | 2.432230 (still oscillating) | 0.826 |
| `nn.MSELoss` + `torch.optim.SGD` | lr = 0.01, 200 epochs | 2.000000 (identical to the manual loop) | 9.0 × 10⁻¹³ |
| Learning rate 0.001 | 200 epochs | 1.977661 | 3.9 × 10⁻³ |
| Learning rate 0.05 | 200 epochs | 2.000000 (loss below 10⁻⁶ by epoch 7) | 0 |

Key findings:

- The five steps of the loop (forward pass, loss, `backward()`, update inside `torch.no_grad()`, `zero_()`) recover the true weight.
- Removing `w.grad.zero_()` makes gradients accumulate. The weight overshoots 2 and keeps oscillating around it with a period of about 16 epochs, so the loss never converges.
- `torch.optim.SGD` and `nn.MSELoss` reproduce the manual loop exactly: same weight and same loss at every epoch.
- For this data each update multiplies the error by $(1 - 15\eta)$, which explains the three learning-rate curves and predicts divergence for $\eta > 2/15 \approx 0.133$.

![Challenge A training loss](images/pytorch_challenge_a_loss.png)

![Loss with and without resetting gradients](images/pytorch_no_zero_grad_comparison.png)

![Loss for three learning rates](images/pytorch_learning_rates.png)

### Challenge B — TensorFlow (John)

*[Placeholder — `tf.GradientTape` results, noise and mini-batch extension, large learning-rate test.]*

### Challenge C — Keras (Eche)

*[Placeholder — `model.fit()` (C1) and custom training loop (C2) results.]*

### Reflection

The 15 reflection questions are answered in the individual notebooks:

| Questions | Notebook |
|---|---|
| Q1–Q5, Q14, Q15 | `GD_sultan_.ipynb` |
| Q6, Q9, Q10 | `GD_Antonio.ipynb` |
| Q7, Q8, Q11 | `GD_John.ipynb` |
| Q12, Q13 | `GD_Eche.ipynb` |

*[Placeholder — link to the merged reflection section after the merge.]*

---

## References

- PyTorch tutorial, *A Gentle Introduction to torch.autograd*: https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- PyTorch documentation, `torch.optim`: https://docs.pytorch.org/docs/stable/optim.html
- TensorFlow guide, *Introduction to gradients and automatic differentiation*: https://www.tensorflow.org/guide/autodiff
- Keras guide, *Writing a training loop from scratch*: https://keras.io/guides/writing_a_training_loop_from_scratch/
- Stanford CS231n: http://cs231n.stanford.edu
- Goodfellow, Bengio and Courville, *Deep Learning*, MIT Press, 2016.
