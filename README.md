# Learning Smoke Removal from Simulation

![Alt Text](https://media.giphy.com/media/2w5MwccaCPBau74nEo/200w_d.gif) Before

![Alt Text](https://media.giphy.com/media/enqo0S1zAClkp7ZoBA/200w_d.gif) After

This repo contains the implementation of CNN based smoke removal from simulation.

If this code is useful for your project, please consider to cite:
```
Chen, L., Wen, T., John, N.W
Unsupervised Learning of Surgical Smoke Removal from Simulation. 
Hamlyn Symposium on Medical Robotics. 2017.
```
### Requirements ###

This code was tested with Blender, Tensorflow, CUDA 8.0 and Ubuntu 16.04.

### Data Preparation ###

We use the [da Vinci dataset](http://hamlyn.doc.ic.ac.uk/vision/data/daVinci.zip) from [Hamlyn Centre Laparoscopic / Endoscopic Video Datasets](http://hamlyn.doc.ic.ac.uk/vision/)

render_smoke.py contains the python script to render smoke on images

Usage:
  1. Modify the data path in render_smoke.py
  2. ./blender --python ./render_smoke.py

### Training ###

desmoke_main.py

### Testing ###

desmoke_test.py
