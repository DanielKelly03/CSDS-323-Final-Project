# CSDS 323 Final Project: CIFAR-10 Model Compression and Evaluation

This project implements and evaluates different compression techniques for deep learning models on the CIFAR-10 dataset. It focuses on applying **SVD-based compression** and **Sparse Low-Rank (SLR)** methods, comparing the performance (accuracy) and compression ratio of each technique.

## Requirements

To run this project, you need to have the following Python packages installed:

- **TensorFlow** (for building, training, and evaluating the model)
- **NumPy** (for matrix manipulations and other numerical operations)
- **Matplotlib** (for generating plots and visualizations)
- **Pandas** (for data handling)
- **Keras** (part of TensorFlow, used for model building)

You can install all the dependencies by running the following command:

```bash
pip install tensorflow==2.18.0 numpy matplotlib pandas
```

Alternatively, you can create a requirements.txt file with the following content:

```bash
tensorflow==2.18.0
numpy
matplotlib
pandas
```

Then, install the dependencies using:

```bash
pip install -r requirements.txt
```

## Setup
1. Clone the Repository

```bash
git clone https://github.com/your-username/CSDS-323-Final-Project.git
cd CSDS-323-Final-Project
```

2. Environment Setup
You can create a virtual environment to isolate the dependencies:

For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

After activating the virtual environment, run pip install -r requirements.txt to install the dependencies.

3. Download the CIFAR-10 Dataset
The dataset will be automatically downloaded by TensorFlow when you run the code. It’s part of the tensorflow.keras.datasets module.

4. Model Files
cifar10vgg.json: This file contains the model architecture (JSON format).
cifar10vgg.weights.h5: This file contains the weights of the trained model (HDF5 format).
If you don't have these files, you'll need to train the model again by running the script train_cifar10vgg.py to generate both files.

5. Training the Model
To train the model and generate the cifar10vgg.json and cifar10vgg.weights.h5 files, run the following command:

```bash
python train_cifar10vgg.py
This will train a VGG-like model on the CIFAR-10 dataset and save both the model's architecture (cifar10vgg.json) and the trained weights (cifar10vgg.weights.h5).
```

6. Running the Compression and Evaluation
Once the model is trained, you can run the experiments using the script:

```bash
python Exp_Cifar10_w.py
```

This script will:

Apply the SVD-based compression to the model’s weight matrices.
Evaluate the model’s performance (accuracy) before and after compression.
Output the compression ratio and accuracy.

7. Plotting Results
To visualize the results of your experiments, run:

```bash
python Plot.py
```

This script will generate bar plots comparing the compression ratio and accuracy for different compression methods and ranks.

**File Descriptions:**
train_cifar10vgg.py: This script trains a VGG-like model on the CIFAR-10 dataset and saves the model architecture and weights.
Exp_Cifar10_w.py: This script applies SVD-based compression to the model and evaluates its performance.
Plot.py: This script generates bar plots comparing compression ratios and accuracies across different methods and ranks.
**Model Details:**
Model Architecture: A VGG-like Convolutional Neural Network (CNN) designed for the CIFAR-10 dataset.
**Compression Techniques:**
SVD (Singular Value Decomposition): A method that reduces the rank of the model’s weight matrices, approximating them with a lower-rank matrix.
SLR (Sparse Low-Rank): A technique that combines sparsity with low-rank approximations for further compression.
**Results:**
The performance of the model and the compression ratio achieved by each technique are reported, including accuracy before and after compression.
The compression ratio is calculated by the number of parameters after compression divided by the number of parameters before compression.

Feel free to reach out if you have any questions or need further assistance in running the project!
