# LGLVAE

#### This is a more user-friendly, Pytorch implementation of the LGL method.

## Installation
### Using uv (recommended)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh             % first install UV
uv python pin 3.13                                          % selects python version to use
uv init newLGL                                              % creates environment folder lglvae
cd newLGL
uv venv                                                     % creates virtual environment
source .venv/bin/activate                                   % activate the environment
uv add git+https://git@github.com/morcoslab/LGL-Pytorch.git   % add package to your environment
```
### Using venv
- First, get a python3 compiler

For Unix:
```bash
python3 -m venv lglvae                                          % creates environment folder lglvae
source lglvae/bin/activate                                      % Activate the environment
pip install git+https://github.com/morcoslab/LGL-Pytorch.git    % creates virtual environment
```

For Windows:
```bash
python3 -m venv lglvae                                          % creates environment folder lglvae
./lglvae/Scripts/activate                                       % Activate the environment
pip install git+https://github.com/morcoslab/LGL-Pytorch.git    % creates virtual environment
```

### Using conda
```bash
conda env create lglvae python=3.13
pip install git+https://github.com/utdal/lgl-vae-pytorch.git

```
## Using the package
To train a model, create the LGL, and plot the landscape with the training data:

```python
from lglvae.lglvae import LGLVAE
import matplotlib.pyplot as plt

save_filename = "model.pkl"
fasta_filename = "your_aligned_sequences.fasta"

# The wrapper class is LGLVAE
model = LGLVAE(fasta_filename)

# Trains a VAE, saves the LGLVAE class at save_filename
model.createVAE(save_filename)

# Creates the mfDCA model, saves the LGLVAE class at save_filename
model.createDCA(save_filename)

# Creates a Hamiltonian landscape with DCA based on where fasta_filename sequences are found in the VAE
model.createLGL(save_filename)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

# Handles plotting the landscape for you, provided an axis
im = model.plot_landscape(ax)
plt.colorbar(im)

# Encodes sequences from a fasta file (here the training sequences, could be any aligned sequences)
sequences = model.encode_sequences(fasta_filename)
ax.scatter(sequences[:, 0], sequences[:, 1], s=3, color="white")

plt.show()
```

At any point the LGLVAE class can be saved, which will hold the VAE model itself, the VAE training logs (loss, reconstruction, kld), the mfDCA parameters, and the landscape file itself.

```python
model.save("some_filename.pkl") # you can save it at any time

model.VAE # this is the VAE model itself
model.VAE.decoder(coordinates) # for instance

loss = model.VAETrainer.training_log["loss"] # or reconstruction, kld
plt.plot(list(range(len(loss))), loss, label="Training Loss")

model.DCA # This is the DCA model (py-mfdca)
model.DCA.compute_Hamiltonians, model.DCA.couplings # for instance

model.LGL # this is the LGL (by default a (500**2, 3) numpy array)
```

This model can be loaded using a load function:
```python
model = LGLVAE.load("path_to_saved_lgl.pkl")
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
model.plot_landscape(ax)
```
You can also sample from the LGL by inputting coordinates. Additionally you can generate samples from the output distributions, instead of taking the maximum probability sequence as was used to create the landscape:
```python
coordinates = torch.randn((100,2))
argmax_sequences = model.generate_sequences(coordinates, argmax_sequence=True)
SeqIO.write(argmax_sequences, "argmax_sequences.fasta", "fasta")
```
## Training on GPUs
This currently should work with CUDA and MPS. For running on clusters with multiple GPUs or where there are more than 1 CUDA device (and you are on a Linux system), you might need to specify a device beforehand:

```bash
export CUDA_VISIBLE_DEVICES=1
python run_training_script.py
```
The number denotes the specific device; you cannot specify more than 1 or you will get an error. I understand this is a Pytorch bug and may not even apply depending on your version. You can specify a device to train on manually with:

```python
model = LGLVAE(fasta_filename)
model.createVAE(save_fn, device="cuda:1") # or whatever device
```

The device is always moved back to CPU for any further inference.
(In the above example "cuda:1" will fail on our HPC system...)

## Using the Bokeh Model Explorer
We built a model plotting script which lets you more easily analyze the landscape. It's included as **bokeh_model_explore.py**, and you will need to install some dependencies before you can use it.

```python
> pip install bokeh scikit-learn scikit-image

> bokeh serve bokeh_model_explore.py

2025-06-24 14:21:29,738 Starting Bokeh server version 3.7.3 (running on Tornado 6.5.1)
2025-06-24 14:21:29,740 User authentication hooks NOT provided (default user enabled)
2025-06-24 14:21:29,742 Bokeh app running at: http://localhost:5006/server
2025-06-24 14:21:29,742 Starting Bokeh server with process id: 89857
```


You can click the http link to open the explorer in the browser.

From here, you can select a folder which contains the model.pkl file, then select the model which you want to load. Similarly you can select an MSA folder and plot sequence in the landscape (**Training MSA dropdown**), and a data folder for sequence labels.

After plotting a fasta set, you can also plot additional fasta files on top of these (with the **Additional Seqs** dropdown box) and they will have coloring which depends on the **"Plotting additional MSA"** button towards the bottom.

For the sequences you plotted with **"Training MSA"**, you can color the plot with the specific label assigned by a csv. The CSV should have a header row, and one row for each sequence in the Training MSA you plot. You can select which column you want to color the sequences with, then use the Legend tab to selectively show/hide sets of sequences based on their label.

Also, with the **Lasso tool**, you can select sequences and it will populate a field on the right with the sequences which you can select and save elsewhere.

## About HMMER and additional tools:
The [py-mfdca][https://github.com/utdal/py-mfdca] module is installed as part of th requirements for the LGL. Some recent tools included in this module use pyhmmer which requires an installation HMMER. If you want to use this set of tools we recommend using UNIX systems or the WLS for Windows. __These tools are not require to use the LGL. If you don't plan to use them, don't worry too much about it.__
