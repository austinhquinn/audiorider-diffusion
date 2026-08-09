# %%
# !! {"metadata":{
# !!   "colab_type": "text",
# !!   "id": "view-in-github"
# !! }}
"""
<a href="https://colab.research.google.com/github/austinhquinn/audiorider-diffusion/blob/main/AudioRider_Diffusion.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# %%
# !! {"metadata":{
# !!   "id": "TitleTop"
# !! }}
"""
# AudioRider (Disco) Diffusion v5.7 - Now with MiDaS (3D mode) not being broken

Disco Diffusion - http://discodiffusion.com/ , https://github.com/alembics/disco-diffusion

In case of confusion, Disco is the name of this notebook edit. The diffusion model in use is Katherine Crowson's fine-tuned 512x512 model

For issues, join the [Disco Diffusion Discord](https://discord.gg/msEZBy4HxA) or message us on twitter at [@somnai_dreams](https://twitter.com/somnai_dreams) or [@gandamu_ml](https://twitter.com/gandamu_ml)
"""

# %%
# !! {"metadata":{
# !!   "id": "CreditsChTop"
# !! }}
"""
### Credits & Changelog ⬇️
"""

# %%
# !! {"metadata":{
# !!   "id": "Credits"
# !! }}
"""
#### Credits

Original notebook by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings). It uses either OpenAI's 256x256 unconditional ImageNet or Katherine Crowson's fine-tuned 512x512 diffusion model (https://github.com/openai/guided-diffusion), together with CLIP (https://github.com/openai/CLIP) to connect text prompts with images.

Modified by Daniel Russell (https://github.com/russelldc, https://twitter.com/danielrussruss) to include (hopefully) optimal params for quick generations in 15-100 timesteps rather than 1000, as well as more robust augmentations.

Further improvements from Dango233 and nshepperd helped improve the quality of diffusion in general, and especially so for shorter runs like this notebook aims to achieve.

Vark added code to load in multiple Clip models at once, which all prompts are evaluated against, which may greatly improve accuracy.

The latest zoom, pan, rotation, and keyframes features were taken from Chigozie Nri's VQGAN Zoom Notebook (https://github.com/chigozienri, https://twitter.com/chigozienri)

Advanced DangoCutn Cutout method is also from Dango223.

--

Disco:

Somnai (https://twitter.com/Somnai_dreams) added Diffusion Animation techniques, QoL improvements and various implementations of tech and techniques, mostly listed in the changelog below.

3D animation implementation added by Adam Letts (https://twitter.com/gandamu_ml) in collaboration with Somnai. Creation of disco.py and ongoing maintenance.

Turbo feature by Chris Allen (https://twitter.com/zippy731)

Improvements to ability to run on local systems, Windows support, and dependency installation by HostsServer (https://twitter.com/HostsServer)

VR Mode by Tom Mason (https://twitter.com/nin_artificial)

Horizontal and Vertical symmetry functionality by nshepperd. Symmetry transformation_steps by huemin (https://twitter.com/huemin_art). Symmetry integration into Disco Diffusion by Dmitrii Tochilkin (https://twitter.com/cut_pow).

Warp and custom model support by Alex Spirin (https://twitter.com/devdef).

Pixel Art Diffusion, Watercolor Diffusion, and Pulp SciFi Diffusion models from KaliYuga (https://twitter.com/KaliYuga_ai). Follow KaliYuga's Twitter for the latest models and for notebooks with specialized settings.

Integration of OpenCLIP models and initiation of integration of KaliYuga models by Palmweaver / Chris Scalf (https://twitter.com/ChrisScalf11)

Integrated portrait_generator_v001 from Felipe3DArtist (https://twitter.com/Felipe3DArtist)
"""

# %%
# !! {"metadata":{
# !!   "id": "LicenseTop"
# !! }}
"""
#### License
"""

# %%
# !! {"metadata":{
# !!   "id": "License"
# !! }}
"""
Licensed under the MIT License

Copyright (c) 2021 Katherine Crowson 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

--

MIT License

Copyright (c) 2019 Intel ISL (Intel Intelligent Systems Lab)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

--

Licensed under the MIT License

Copyright (c) 2021 Maxwell Ingham

Copyright (c) 2022 Adam Letts 

Copyright (c) 2022 Alex Spirin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

--
flow-related - https://github.com/NVIDIA/flownet2-pytorch/blob/master/LICENSE
--
Copyright 2017 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# %%
# !! {"metadata":{
# !!   "id": "ChangelogTop"
# !! }}
"""
#### Changelog
"""

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "Changelog"
# !! }}
#@title <- View Changelog
skip_for_run_all = True #@param {type: 'boolean'}

if skip_for_run_all == False:
  print(
      '''
  v1 Update: Oct 29th 2021 - Somnai

      QoL improvements added by Somnai (@somnai_dreams), including user friendly UI, settings+prompt saving and improved google drive folder organization.

  v1.1 Update: Nov 13th 2021 - Somnai

      Now includes sizing options, intermediate saves and fixed image prompts and perlin inits. unexposed batch option since it doesn't work

  v2 Update: Nov 22nd 2021 - Somnai

      Initial addition of Katherine Crowson's Secondary Model Method (https://colab.research.google.com/drive/1mpkrhOjoyzPeSWy2r7T8EYRaU7amYOOi#scrollTo=X5gODNAMEUCR)

      Noticed settings were saving with the wrong name so corrected it. Let me know if you preferred the old scheme.

  v3 Update: Dec 24th 2021 - Somnai

      Implemented Dango's advanced cutout method

      Added SLIP models, thanks to NeuralDivergent

      Fixed issue with NaNs resulting in black images, with massive help and testing from @Softology

      Perlin now changes properly within batches (not sure where this perlin_regen code came from originally, but thank you)

  v4 Update: Jan 2022 - Somnai

      Implemented Diffusion Zooming

      Added Chigozie keyframing

      Made a bunch of edits to processes
  
  v4.1 Update: Jan 14th 2022 - Somnai

      Added video input mode

      Added license that somehow went missing

      Added improved prompt keyframing, fixed image_prompts and multiple prompts

      Improved UI

      Significant under the hood cleanup and improvement

      Refined defaults for each mode

      Added latent-diffusion SuperRes for sharpening

      Added resume run mode

  v4.9 Update: Feb 5th 2022 - gandamu / Adam Letts

      Added 3D

      Added brightness corrections to prevent animation from steadily going dark over time

  v4.91 Update: Feb 19th 2022 - gandamu / Adam Letts

      Cleaned up 3D implementation and made associated args accessible via Colab UI elements

  v4.92 Update: Feb 20th 2022 - gandamu / Adam Letts

      Separated transform code

  v5.01 Update: Mar 10th 2022 - gandamu / Adam Letts

      IPython magic commands replaced by Python code

  v5.1 Update: Mar 30th 2022 - zippy / Chris Allen and gandamu / Adam Letts

      Integrated Turbo+Smooth features from Disco Diffusion Turbo -- just the implementation, without its defaults.

      Implemented resume of turbo animations in such a way that it's now possible to resume from different batch folders and batch numbers.

      3D rotation parameter units are now degrees (rather than radians)

      Corrected name collision in sampling_mode (now diffusion_sampling_mode for plms/ddim, and sampling_mode for 3D transform sampling)

      Added video_init_seed_continuity option to make init video animations more continuous

  v5.1 Update: Apr 4th 2022 - MSFTserver aka HostsServer

      Removed pytorch3d from needing to be compiled with a lite version specifically made for Disco Diffusion

      Remove Super Resolution

      Remove SLIP Models

      Update for crossplatform support

  v5.2 Update: Apr 10th 2022 - nin_artificial / Tom Mason

      VR Mode

  v5.3 Update: Jun 10th 2022 - nshepperd, huemin, cut_pow / Dmitrii Tochilkin

      Horizontal and Vertical symmetry

      Addition of ViT-L/14@336px model (requires high VRAM)

  v5.4 Update: Jun 14th 2022 - devdef / Alex Spirin, Alex's Warp changes integrated into DD main by gandamu / Adam Letts

      Warp mode - for smooth/continuous video input results leveraging optical flow estimation and frame blending

      Custom models support

  v5.5 Update: Jul 11th 2022 - Palmweaver / Chris Scalf, KaliYuga_ai, further DD integration by gandamu / Adam Letts

      OpenCLIP models integration

      Pixel Art Diffusion, Watercolor Diffusion, and Pulp SciFi Diffusion models

      cut_ic_pow scheduling

  v5.6 Update: Jul 13th 2022 - Felipe3DArtist integration by gandamu / Adam Letts

      portrait_generator_v001 diffusion model integrated

  v5.61 Update: Aug 21st 2022 - gandamu / Adam Letts

      Correct progress bars issue caused by recent Google Colab environment changes
      Adjust 512x512 diffusion model download URI priority due to issues with the previous primary source

  v5.7 Update: Dec 31st 2022 - Steffen Moelter (with minor colab-convert integration by gandamu)

      Clone MiDaS v3 specifically. This fixes 3D mode. It had been broken since MiDaS v3.1 introduced an incompatibility.
    '''
  )

# %%
# !! {"metadata":{
# !!   "id": "TutorialTop"
# !! }}
"""
# Tutorial
"""

# %%
# !! {"metadata":{
# !!   "id": "DiffusionSet"
# !! }}
"""
**Diffusion settings (Defaults are heavily outdated)**
---
Disco Diffusion is complex, and continually evolving with new features.  The most current documentation on on Disco Diffusion settings can be found in the unofficial guidebook:

[Zippy's Disco Diffusion Cheatsheet](https://docs.google.com/document/d/1l8s7uS2dGqjztYSjPpzlmXLjl5PM3IGkRWI3IiCuK7g/edit)

We also encourage users to join the [Disco Diffusion User Discord](https://discord.gg/XGZrFFCRfN) to learn from the active user community.

This section below is outdated as of v2

Setting | Description | Default
--- | --- | ---
**Your vision:**
`text_prompts` | A description of what you'd like the machine to generate. Think of it like writing the caption below your image on a website. | N/A
`image_prompts` | Think of these images more as a description of their contents. | N/A
**Image quality:**
`clip_guidance_scale`  | Controls how much the image should look like the prompt. | 1000
`tv_scale` | Controls the smoothness of the final output. | 150
`range_scale` | Controls how far out of range RGB values are allowed to be. | 150
`sat_scale` | Controls how much saturation is allowed. From nshepperd's JAX notebook. | 0
`cutn` | Controls how many crops to take from the image. | 16
`cutn_batches` | Accumulate CLIP gradient from multiple batches of cuts. | 2
**Init settings:**
`init_image` | URL or local path | None
`init_scale` | This enhances the effect of the init image, a good value is 1000 | 0
`skip_steps` | Controls the starting point along the diffusion timesteps | 0
`perlin_init` | Option to start with random perlin noise | False
`perlin_mode` | ('gray', 'color') | 'mixed'
**Advanced:**
`skip_augs` | Controls whether to skip torchvision augmentations | False
`randomize_class` | Controls whether the imagenet class is randomly changed each iteration | True
`clip_denoised` | Determines whether CLIP discriminates a noisy or denoised image | False
`clamp_grad` | Experimental: Using adaptive clip grad in the cond_fn | True
`seed`  | Choose a random seed and print it at end of run for reproduction | random_seed
`fuzzy_prompt` | Controls whether to add multiple noisy prompts to the prompt losses | False
`rand_mag` | Controls the magnitude of the random noise | 0.1
`eta` | DDIM hyperparameter | 0.5
`use_vertical_symmetry` | Enforce symmetry over x axis of the image on [`tr_st`*`steps` for `tr_st` in `transformation_steps`] steps of the diffusion process | False
`use_horizontal_symmetry` | Enforce symmetry over y axis of the image on [`tr_st`*`steps` for `tr_st` in `transformation_steps`] steps of the diffusion process | False
`transformation_steps` | Steps (expressed in percentages) in which the symmetry is enforced | [0.01]
`video_init_flow_warp` | Flow warp enabled | True
`video_init_flow_blend` | 0 - you get raw input, 1 - you get warped diffused previous frame  | 0.999
`video_init_check_consistency` | TBD check forward-backward flow consistency (uncheck unless there are too many warping artifacts) | False

..

**Model settings**
---

Setting | Description | Default
--- | --- | ---
**Diffusion:**
`timestep_respacing` | Modify this value to decrease the number of timesteps. | ddim100
`diffusion_steps` || 1000
**Diffusion:**
`clip_models` | Models of CLIP to load. Typically the more, the better but they all come at a hefty VRAM cost. | ViT-B/32, ViT-B/16, RN50x4
"""

# %%
# !! {"metadata":{
# !!   "id": "SetupTop"
# !! }}
"""
# 1. Set Up
"""

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "CheckGPU"
# !! }}
#@title 1.1 Check GPU Status
import subprocess
simple_nvidia_smi_display = False#@param {type:"boolean"}
if simple_nvidia_smi_display:
    #!nvidia-smi
    nvidiasmi_output = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(nvidiasmi_output)
else:
    #!nvidia-smi -i 0 -e 0
    nvidiasmi_output = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(nvidiasmi_output)
    nvidiasmi_ecc_note = subprocess.run(['nvidia-smi', '-i', '0', '-e', '0'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(nvidiasmi_ecc_note)

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "PrepFolders"
# !! }}
#@title 1.2 Prepare Folders
import subprocess, os, sys, ipykernel

def gitclone(url, target_dir=None, branch_arg=None):
    run_args = ['git', 'clone']
    if branch_arg:
        run_args.extend(['-b', branch_arg])
    run_args.append(url)
    if target_dir:
        run_args.append(target_dir)
    res = subprocess.run(run_args, stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def pipi(modulestr):
    res = subprocess.run(['pip', 'install', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def pipie(modulestr):
    res = subprocess.run(['git', 'install', '-e', modulestr], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(res)

def wget(url, outputdir):
    # The original shelled out to a system `wget` binary, which Colab/Linux
    # always has and Windows never does -- every model download in this
    # notebook went through this function, so a missing `wget` on Windows
    # surfaced later as a confusing "file not found" for whatever checkpoint
    # was supposedly being downloaded. urlretrieve is stdlib, follows
    # redirects the same way curl -L does, and works on every platform.
    import urllib.request
    from urllib.parse import urlparse

    filename = os.path.basename(urlparse(url).path)
    dest = os.path.join(outputdir, filename)
    createPath(outputdir)

    def _progress(block_num, block_size, total_size):
        if total_size <= 0 or block_num % 200:
            return
        downloaded = min(block_num * block_size, total_size)
        print(f'{filename}: {downloaded * 100 // total_size}% '
              f'({downloaded // (1024 * 1024)}MB / {total_size // (1024 * 1024)}MB)')

    print(f'Downloading {url} -> {dest}')
    urllib.request.urlretrieve(url, dest, reporthook=_progress)

try:
    from google.colab import drive
    print("Google Colab detected. Using Google Drive.")
    is_colab = True
    #@markdown If you connect your Google Drive, you can save the final image of each run on your drive.
    google_drive = True #@param {type:"boolean"}
    #@markdown Click here if you'd like to save the diffusion model checkpoint file to (and/or load from) your Google Drive:
    save_models_to_google_drive = True #@param {type:"boolean"}
    print("Downgrading ipywidgets to latest 7.x in order to enable custom widget manager (for tqdm progress bars)")
    multipip_res = subprocess.run(['pip', 'install', 'ipywidgets>=7,<8'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(multipip_res)
    from google.colab import output
    output.enable_custom_widget_manager()
except:
    is_colab = False
    google_drive = False
    save_models_to_google_drive = False
    print("Google Colab not detected.")

if is_colab:
    if google_drive is True:
        drive.mount('/content/drive')
        root_path = '/content/drive/MyDrive/AI/Disco_Diffusion'
    else:
        root_path = '/content'
else:
    root_path = os.getcwd()

import os
def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)

initDirPath = f'{root_path}/init_images'
createPath(initDirPath)
outDirPath = f'{root_path}/images_out'
createPath(outDirPath)

if is_colab:
    if google_drive and not save_models_to_google_drive or not google_drive:
        model_path = '/content/models'
        createPath(model_path)
    if google_drive and save_models_to_google_drive:
        model_path = f'{root_path}/models'
        createPath(model_path)
else:
    model_path = f'{root_path}/models'
    createPath(model_path)

# libraries = f'{root_path}/libraries'
# createPath(libraries)

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "InstallDeps"
# !! }}
#@title ### 1.3 Install, import dependencies and set up runtime devices

import pathlib, shutil, os, sys

# There are some reports that with a T4 or V100 on Colab, downgrading to a previous version of PyTorch may be necessary.
# .. but there are also reports that downgrading breaks them!  If you're facing issues, you may want to try uncommenting and running this code.
# nvidiasmi_output = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
# cards_requiring_downgrade = ["Tesla T4", "V100"]
# if is_colab:
#     if any(cardstr in nvidiasmi_output for cardstr in cards_requiring_downgrade):
#         print("Downgrading pytorch. This can take a couple minutes ...")
#         downgrade_pytorch_result = subprocess.run(['pip', 'install', 'torch==1.10.2', 'torchvision==0.11.3', '-q'], stdout=subprocess.PIPE).stdout.decode('utf-8')
#         print("pytorch downgraded.")

#@markdown Check this if you want to use CPU
useCPU = False #@param {type:"boolean"}

if not is_colab:
    # If running locally, there's a good chance your env will need this in order to not crash upon np.matmul() or similar operations.
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

PROJECT_DIR = os.path.abspath(os.getcwd())
# AdaBins' checkpoint host (cloudflare-ipfs.com) has been dead since Aug 2024,
# and InferenceHelper/MAX_ADABINS_AREA (the only names this setup produces)
# are never referenced anywhere outside this one gated block -- disabling it
# has no downstream effect. See MIGRATION.md blocker #3.
USE_ADABINS = False

if is_colab:
    if not google_drive:
        root_path = f'/content'
        model_path = '/content/models' 
else:
    root_path = os.getcwd()
    model_path = f'{root_path}/models'

multipip_res = subprocess.run(['pip', 'install', 'lpips', 'datetime', 'timm', 'ftfy', 'einops', 'pytorch-lightning', 'omegaconf', 'transformers'], stdout=subprocess.PIPE).stdout.decode('utf-8')
print(multipip_res)

if is_colab:
    subprocess.run(['apt', 'install', 'imagemagick'], stdout=subprocess.PIPE).stdout.decode('utf-8')

try:
    from CLIP import clip
except:
    if not os.path.exists("CLIP"):
        gitclone("https://github.com/openai/CLIP")
    sys.path.append(f'{PROJECT_DIR}/CLIP')

try:
    import open_clip
except:
    if not os.path.exists("open_clip/src"):
        gitclone("https://github.com/mlfoundations/open_clip.git")
    sys.path.append(f'{PROJECT_DIR}/open_clip/src')
    import open_clip

try:
    from guided_diffusion.script_util import create_model_and_diffusion
except:
    if not os.path.exists("guided-diffusion"):
        gitclone("https://github.com/kostarion/guided-diffusion")
    sys.path.append(f'{PROJECT_DIR}/guided-diffusion')

try:
    from resize_right import resize
except:
    if not os.path.exists("ResizeRight"):
        gitclone("https://github.com/assafshocher/ResizeRight.git")
    sys.path.append(f'{PROJECT_DIR}/ResizeRight')

try:
    import py3d_tools
except:
    if not os.path.exists('pytorch3d-lite'):
        gitclone("https://github.com/MSFTserver/pytorch3d-lite.git")
    sys.path.append(f'{PROJECT_DIR}/pytorch3d-lite')

try:
    from midas.dpt_depth import DPTDepthModel
except:
    if not os.path.exists('MiDaS'):
        gitclone("https://github.com/isl-org/MiDaS.git", branch_arg="v3")
    if not os.path.exists('MiDaS/midas_utils.py'):
        shutil.move('MiDaS/utils.py', 'MiDaS/midas_utils.py')
    if not os.path.exists(f'{model_path}/dpt_large-midas-2f21e586.pt'):
        wget("https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt", model_path)
    sys.path.append(f'{PROJECT_DIR}/MiDaS')

try:
    sys.path.append(PROJECT_DIR)
    import disco_xform_utils as dxf
except:
    if not os.path.exists("disco-diffusion"):
        gitclone("https://github.com/alembics/disco-diffusion.git")
    if not os.path.exists('disco_xform_utils.py'):
        shutil.move('disco-diffusion/disco_xform_utils.py', 'disco_xform_utils.py')
    sys.path.append(PROJECT_DIR)

import torch
from dataclasses import dataclass
from functools import partial
import cv2
import pandas as pd
import gc
import io
import math
import timm
from IPython import display
import lpips
from PIL import Image, ImageOps
import requests
from glob import glob
import json
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
from CLIP import clip
from resize_right import resize
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from ipywidgets import Output
import hashlib
from functools import partial
if is_colab:
    os.chdir('/content')
    from google.colab import files
else:
    os.chdir(f'{PROJECT_DIR}')
from IPython.display import Image as ipyimg
from numpy import asarray
from einops import rearrange, repeat
import torch, torchvision
import time
from omegaconf import OmegaConf
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# AdaBins stuff
if USE_ADABINS:
    try:
        from infer import InferenceHelper
    except:
        if not os.path.exists("AdaBins"):
            gitclone("https://github.com/shariqfarooq123/AdaBins.git")
        if not os.path.exists(f'{PROJECT_DIR}/pretrained/AdaBins_nyu.pt'):
            createPath(f'{PROJECT_DIR}/pretrained')
            wget("https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt", f'{PROJECT_DIR}/pretrained')
        sys.path.append(f'{PROJECT_DIR}/AdaBins')
    from infer import InferenceHelper
    MAX_ADABINS_AREA = 500000

import torch
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not useCPU) else 'cpu')
print('Using device:', DEVICE)
device = DEVICE # At least one of the modules expects this name..

if not useCPU:
    if torch.cuda.get_device_capability(DEVICE) == (8,0): ## A100 fix thanks to Emad
        print('Disabling CUDNN for A100 gpu', file=sys.stderr)
        torch.backends.cudnn.enabled = False

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "ybXXmTilwViA"
# !! }}
#@title ### 1.3b Install Pydub and import Audio dependencies

!pip install pydub
!pip install scipy
!pip install sci-analysis
!pip install easing-functions librosa 
!pip install scikit-image
#from google.colab import widgets
import ipywidgets
#from google.colab.display import display, clear_output
%matplotlib inline 
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython import display
import scipy.stats as st
# sci_analysis.analyze was imported but never called anywhere in this file,
# and the package itself doesn't import under NumPy 2.x (uses the removed
# np.float_ alias) -- removed rather than fought, since it was dead code.
#from easing_functions import *
import numpy as np
from pydub import AudioSegment
import pydub.scipy_effects 
import librosa 
import librosa.display
import skimage.io
from PIL import Image, ImageDraw, ImageFont

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "uLfrsE_yZi5T"
# !! }}
#@title ### 1.3c Install audio vocals to text prompt (future use)

splitaudio=False #@param {'type': 'boolean'}




if splitaudio:
  @unique
  class ASREngine(Enum):
    sphinx = 0
    google = 1
  def speech_to_text(filename: str, engine: ASREngine, language: str, show_all: bool = False) -> str:
    r = sr.Recognizer()

    with sr.AudioFile(filename) as source:
      audio = r.record(source)

      asr_functions = {
        ASREngine.sphinx: r.recognize_sphinx,
        ASREngine.google: r.recognize_google,
      }

      response = asr_functions[engine](audio, language=language, show_all=show_all)
      return response

  %shell apt-get install -y ffmpeg swig libpulse-dev
  %shell swig -version
  %shell pip3 install pocketsphinx SpeechRecognition
  %shell pip install spleeter
  import speech_recognition as sr
  from enum import Enum, unique
  %shell spleeter separate -o /content/input/  /content/input/input.mp3
  # Audio('/content/input/input/vocals.wav')
  # Audio('/content/input/input/accompaniment.wav')
  vocal_filename = '/content/input/input/vocals.wav'
  lyric_filename = '/content/input/lyrics.txt'
  lang = 'en-US'
  #print('\naudio file="{0}"    expected text="{1}"'.format(filename, text))
  for asr_engine in ASREngine:
    try:
      response = speech_to_text(vocal_filename, asr_engine, language=lang)
      print('{0}: "{1}"'.format(asr_engine.name, response))
    except sr.UnknownValueError:
      print('{0} could not understand audio'.format(asr_engine.name))
    except sr.RequestError as e:
      print('{0} error: {0}'.format(asr_engine.name, e))

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "DefMidasFns"
# !! }}
#@title ### 1.4 Define Midas functions

from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

# Initialize MiDaS depth model.
# It remains resident in VRAM and likely takes around 2GB VRAM.
# You could instead initialize it for each frame (and free it after each frame) to save VRAM.. but initializing it is slow.
default_models = {
    "midas_v21_small": f"{model_path}/midas_v21_small-70d6b9c8.pt",
    "midas_v21": f"{model_path}/midas_v21-f6b98070.pt",
    "dpt_large": f"{model_path}/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": f"{model_path}/dpt_hybrid-midas-501f0c75.pt",
    "dpt_hybrid_nyu": f"{model_path}/dpt_hybrid_nyu-2ce69ec7.pt",}


def init_midas_depth_model(midas_model_type="dpt_large", optimize=True):
    midas_model = None
    net_w = None
    net_h = None
    resize_mode = None
    normalization = None

    print(f"Initializing MiDaS '{midas_model_type}' depth model...")
    # load network
    midas_model_path = default_models[midas_model_type]

    if midas_model_type == "dpt_large": # DPT-Large
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "dpt_hybrid": #DPT-Hybrid
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "dpt_hybrid_nyu": #DPT-Hybrid-NYU
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "midas_v21":
        midas_model = MidasNet(midas_model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif midas_model_type == "midas_v21_small":
        midas_model = MidasNet_small(midas_model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"midas_model_type '{midas_model_type}' not implemented")
        assert False

    midas_transform = T.Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    midas_model.eval()
    
    if optimize==True:
        if DEVICE == torch.device("cuda"):
            midas_model = midas_model.to(memory_format=torch.channels_last)  
            midas_model = midas_model.half()

    midas_model.to(DEVICE)

    print(f"MiDaS '{midas_model_type}' depth model initialized.")
    return midas_model, midas_transform, net_w, net_h, resize_mode, normalization

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "DefFns"
# !! }}
#@title 1.5 Define necessary functions

# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869

import py3d_tools as p3dT
import disco_xform_utils as dxf

def interp(t):
    return 3 * t**2 - 2 * t ** 3

def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)

def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0]//3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out

def regen_perlin():
    if perlin_mode == 'color':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size, -1, -1, -1)

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()

def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2]//4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn//4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size/max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts

cutout_debug = False
padargs = {}

class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size,
                 Overview=4, 
                 InnerCrop = 0, IC_Size_Pow=0.5, IC_Grey_P = 0.2
                 ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        if args.animation_mode == 'None':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.5),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.1),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
          ])
        elif args.animation_mode == 'Video Input':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.5),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomPerspective(distortion_scale=0.4, p=0.7),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.15),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
          ])
        elif  args.animation_mode == '2D' or args.animation_mode == '3D':
          self.augs = T.Compose([
              T.RandomHorizontalFlip(p=0.4),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomAffine(degrees=10, translate=(0.05, 0.05),  interpolation = T.InterpolationMode.BILINEAR),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.RandomGrayscale(p=0.1),
              T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
              T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
          ])
          

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1,3,self.cut_size,self.cut_size] 
        output_shape_2 = [1,3,self.cut_size+2,self.cut_size+2]
        pad_input = F.pad(input,((sideY-max_size)//2,(sideY-max_size)//2,(sideX-max_size)//2,(sideX-max_size)//2), **padargs)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview>0:
            if self.Overview<=4:
                if self.Overview>=1:
                    cutouts.append(cutout)
                if self.Overview>=2:
                    cutouts.append(gray(cutout))
                if self.Overview>=3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview==4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if cutout_debug:
                if is_colab:
                    TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("/content/cutout_overview0.jpg",quality=99)
                else:
                    TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("cutout_overview0.jpg",quality=99)

                              
        if self.InnerCrop >0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([])**self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                if is_colab:
                    TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("/content/cutout_InnerCrop.jpg",quality=99)
                else:
                    TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("cutout_InnerCrop.jpg",quality=99)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True: cutouts=self.augs(cutouts)
        return cutouts

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)     

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0/200.0

def do_3d_step(img_filepath, frame_num, midas_model, midas_transform):
  if args.key_frames:
    translation_x = args.translation_x_series[frame_num]
    translation_y = args.translation_y_series[frame_num]
    translation_z = args.translation_z_series[frame_num]
    rotation_3d_x = args.rotation_3d_x_series[frame_num]
    rotation_3d_y = args.rotation_3d_y_series[frame_num]
    rotation_3d_z = args.rotation_3d_z_series[frame_num]
    print(
        f'translation_x: {translation_x}',
        f'translation_y: {translation_y}',
        f'translation_z: {translation_z}',
        f'rotation_3d_x: {rotation_3d_x}',
        f'rotation_3d_y: {rotation_3d_y}',
        f'rotation_3d_z: {rotation_3d_z}',
    )

  translate_xyz = [-translation_x*TRANSLATION_SCALE, translation_y*TRANSLATION_SCALE, -translation_z*TRANSLATION_SCALE]
  rotate_xyz_degrees = [rotation_3d_x, rotation_3d_y, rotation_3d_z]
  print('translation:',translate_xyz)
  print('rotation:',rotate_xyz_degrees)
  rotate_xyz = [math.radians(rotate_xyz_degrees[0]), math.radians(rotate_xyz_degrees[1]), math.radians(rotate_xyz_degrees[2])]
  rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
  print("rot_mat: " + str(rot_mat))
  next_step_pil = dxf.transform_image_3d(img_filepath, midas_model, midas_transform, DEVICE,
                                          rot_mat, translate_xyz, args.near_plane, args.far_plane,
                                          args.fov, padding_mode=args.padding_mode,
                                          sampling_mode=args.sampling_mode, midas_weight=args.midas_weight)
  return next_step_pil

def symmetry_transformation_fn(x):
    if args.use_horizontal_symmetry:
        [n, c, h, w] = x.size()
        x = torch.concat((x[:, :, :, :w//2], torch.flip(x[:, :, :, :w//2], [-1])), -1)
        print("horizontal symmetry applied")
    if args.use_vertical_symmetry:
        [n, c, h, w] = x.size()
        x = torch.concat((x[:, :, :h//2, :], torch.flip(x[:, :, :h//2, :], [-2])), -2)
        print("vertical symmetry applied")
    return x

def do_run():
  seed = args.seed
  print(range(args.start_frame, args.max_frames))

  if (args.animation_mode == "3D") and (args.midas_weight > 0.0):
      midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args.midas_depth_model)
  for frame_num in range(args.start_frame, args.max_frames):
      if stop_on_next_loop:
        break
      
      display.clear_output(wait=True)

      # Print Frame progress if animation mode is on
      if args.animation_mode != "None":
        batchBar = tqdm(range(args.max_frames), desc ="Frames")
        batchBar.n = frame_num
        batchBar.refresh()

      
      # Inits if not video frames
      if args.animation_mode != "Video Input":
        if args.init_image in ['','none', 'None', 'NONE']:
          init_image = None
        else:
          init_image = args.init_image
        init_scale = args.init_scale
        skip_steps = args.skip_steps

      if args.animation_mode == "2D":
        if args.key_frames:
          angle = args.angle_series[frame_num]
          zoom = args.zoom_series[frame_num]
          translation_x = args.translation_x_series[frame_num]
          translation_y = args.translation_y_series[frame_num]
          print(
              f'angle: {angle}',
              f'zoom: {zoom}',
              f'translation_x: {translation_x}',
              f'translation_y: {translation_y}',
          )
        
        if frame_num > 0:
          seed += 1
          if resume_run and frame_num == start_frame:
            img_0 = cv2.imread(batchFolder+f"/{batch_name}({batchNum})_{start_frame-1:04}.png")
          else:
            img_0 = cv2.imread('prevFrame.png')
          center = (1*img_0.shape[1]//2, 1*img_0.shape[0]//2)
          trans_mat = np.float32(
              [[1, 0, translation_x],
              [0, 1, translation_y]]
          )
          rot_mat = cv2.getRotationMatrix2D( center, angle, zoom )
          trans_mat = np.vstack([trans_mat, [0,0,1]])
          rot_mat = np.vstack([rot_mat, [0,0,1]])
          transformation_matrix = np.matmul(rot_mat, trans_mat)
          img_0 = cv2.warpPerspective(
              img_0,
              transformation_matrix,
              (img_0.shape[1], img_0.shape[0]),
              borderMode=cv2.BORDER_WRAP
          )

          cv2.imwrite('prevFrameScaled.png', img_0)
          init_image = 'prevFrameScaled.png'
          init_scale = args.frames_scale
          skip_steps = args.calc_frames_skip_steps

      if args.animation_mode == "3D":
        if frame_num > 0:
          seed += 1    
          if resume_run and frame_num == start_frame:
            img_filepath = batchFolder+f"/{batch_name}({batchNum})_{start_frame-1:04}.png"
            if turbo_mode and frame_num > turbo_preroll:
              shutil.copyfile(img_filepath, 'oldFrameScaled.png')
          else:
            img_filepath = '/content/prevFrame.png' if is_colab else 'prevFrame.png'

          next_step_pil = do_3d_step(img_filepath, frame_num, midas_model, midas_transform)
          next_step_pil.save('prevFrameScaled.png')

          ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
          if turbo_mode:
            if frame_num == turbo_preroll: #start tracking oldframe
              next_step_pil.save('oldFrameScaled.png')#stash for later blending          
            elif frame_num > turbo_preroll:
              #set up 2 warped image sequences, old & new, to blend toward new diff image
              old_frame = do_3d_step('oldFrameScaled.png', frame_num, midas_model, midas_transform)
              old_frame.save('oldFrameScaled.png')
              if frame_num % int(turbo_steps) != 0: 
                print('turbo skip this frame: skipping clip diffusion steps')
                filename = f'{args.batch_name}({args.batchNum})_{frame_num:04}.png'
                blend_factor = ((frame_num % int(turbo_steps))+1)/int(turbo_steps)
                print('turbo skip this frame: skipping clip diffusion steps and saving blended frame')
                newWarpedImg = cv2.imread('prevFrameScaled.png')#this is already updated..
                oldWarpedImg = cv2.imread('oldFrameScaled.png')
                blendedImage = cv2.addWeighted(newWarpedImg, blend_factor, oldWarpedImg,1-blend_factor, 0.0)
                cv2.imwrite(f'{batchFolder}/{filename}',blendedImage)
                next_step_pil.save(f'{img_filepath}') # save it also as prev_frame to feed next iteration
                if vr_mode:
                  generate_eye_views(TRANSLATION_SCALE,batchFolder,filename,frame_num,midas_model, midas_transform)
                continue
              else:
                #if not a skip frame, will run diffusion and need to blend.
                oldWarpedImg = cv2.imread('prevFrameScaled.png')
                cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later 
                print('clip/diff this frame - generate clip diff image')

          init_image = 'prevFrameScaled.png'
          init_scale = args.frames_scale
          skip_steps = args.calc_frames_skip_steps

      if  args.animation_mode == "Video Input":
        init_scale = args.video_init_frames_scale
        skip_steps = args.calc_frames_skip_steps
        if not video_init_seed_continuity:
          seed += 1
        if video_init_flow_warp:
          if frame_num == 0: 
            skip_steps = args.video_init_skip_steps
            init_image = f'{videoFramesFolder}/{frame_num+1:04}.jpg'
          if frame_num > 0: 
            prev = PIL.Image.open(batchFolder+f"/{batch_name}({batchNum})_{frame_num-1:04}.png")
            
            frame1_path = f'{videoFramesFolder}/{frame_num:04}.jpg'
            frame2 = PIL.Image.open(f'{videoFramesFolder}/{frame_num+1:04}.jpg')
            flo_path = f"/{flo_folder}/{frame1_path.split('/')[-1]}.npy"
            
            init_image = 'warped.png'
            print(video_init_flow_blend)
            weights_path = None
            if video_init_check_consistency:
                # TBD
                pass
 
            warp(prev, frame2, flo_path, blend=video_init_flow_blend, weights_path=weights_path).save(init_image)
            
        else:
          init_image = f'{videoFramesFolder}/{frame_num+1:04}.jpg'


      loss_values = []
  
      if seed is not None:
          np.random.seed(seed)
          random.seed(seed)
          torch.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
          torch.backends.cudnn.deterministic = True
  
      target_embeds, weights = [], []
      
      if args.prompts_series is not None and frame_num >= len(args.prompts_series):
        frame_prompt = args.prompts_series[-1]
      elif args.prompts_series is not None:
        frame_prompt = args.prompts_series[frame_num]
      else:
        frame_prompt = []
      
      print(args.image_prompts_series)
      if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
        image_prompt = args.image_prompts_series[-1]
      elif args.image_prompts_series is not None:
        image_prompt = args.image_prompts_series[frame_num]
      else:
        image_prompt = []

      print(f'Frame {frame_num} Prompt: {frame_prompt}')

      model_stats = []
      for clip_model in clip_models:
            cutn = 16
            model_stat = {"clip_model":None,"target_embeds":[],"make_cutouts":None,"weights":[]}
            model_stat["clip_model"] = clip_model
            
            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
                
                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append((txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(0,1))
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)
        
            if image_prompt:
              model_stat["make_cutouts"] = MakeCutouts(clip_model.visual.input_resolution, cutn, skip_augs=skip_augs) 
              for prompt in image_prompt:
                  path, weight = parse_prompt(prompt)
                  img = Image.open(fetch(path)).convert('RGB')
                  img = TF.resize(img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS)
                  batch = model_stat["make_cutouts"](TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
                  embed = clip_model.encode_image(normalize(batch)).float()
                  if fuzzy_prompt:
                      for i in range(25):
                          model_stat["target_embeds"].append((embed + torch.randn(embed.shape).cuda() * rand_mag).clamp(0,1))
                          weights.extend([weight / cutn] * cutn)
                  else:
                      model_stat["target_embeds"].append(embed)
                      model_stat["weights"].extend([weight / cutn] * cutn)
        
            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)
  
      init = None
      if init_image is not None:
          init = Image.open(fetch(init_image)).convert('RGB')
          init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
          init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
      
      if args.perlin_init:
          if args.perlin_mode == 'color':
              init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
              init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
          elif args.perlin_mode == 'gray':
            init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
            init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
          else:
            init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
            init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
          # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
          init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
          del init2
  
      cur_t = None
  
      def cond_fn(x, t, y=None):
          with torch.enable_grad():
              x_is_NaN = False
              x = x.detach().requires_grad_()
              n = x.shape[0]
              if use_secondary_model is True:
                alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                cosine_t = alpha_sigma_to_t(alpha, sigma)
                out = secondary_model(x, cosine_t[None].repeat([n])).pred
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
              else:
                my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out['pred_xstart'] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
              for model_stat in model_stats:
                for i in range(args.cutn_batches):
                    t_int = int(t.item())+1 #errors on last step without +1, need to find source
                    #when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                    try:
                        input_resolution=model_stat["clip_model"].visual.input_resolution
                    except:
                        input_resolution=224
## cut pow and inner/outer
                    cuts = MakeCutoutsDango(input_resolution,
                            Overview= args.cut_overview[1000-t_int], 
                            InnerCrop = args.cut_innercut[1000-t_int],
                            IC_Size_Pow=args.cut_ic_pow[1000-t_int],
                            IC_Grey_P = args.cut_icgray_p[1000-t_int]
                            )
                    clip_in = normalize(cuts(x_in.add(1).div(2)))
                    image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                    dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                    dists = dists.view([args.cut_overview[1000-t_int]+args.cut_innercut[1000-t_int], n, -1])
                    losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                    loss_values.append(losses.sum().item()) # log loss, probably shouldn't do per cutn_batch
                    x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
              tv_losses = tv_loss(x_in)
              if use_secondary_model is True:
                range_losses = range_loss(out)
              else:
                range_losses = range_loss(out['pred_xstart'])
              sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
              loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale
              if init is not None and init_scale:
                  init_losses = lpips_model(x_in, init)
                  loss = loss + init_losses.sum() * init_scale
              x_in_grad += torch.autograd.grad(loss, x_in)[0]
              if torch.isnan(x_in_grad).any()==False:
                  grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
              else:
                # print("NaN'd")
                x_is_NaN = True
                grad = torch.zeros_like(x)
          if args.clamp_grad and x_is_NaN == False:
              magnitude = grad.square().mean().sqrt()
              return grad * magnitude.clamp(max=args.clamp_max) / magnitude  #min=-0.02, min=-clamp_max, 
          return grad
  
      if args.diffusion_sampling_mode == 'ddim':
          sample_fn = diffusion.ddim_sample_loop_progressive
      else:
          sample_fn = diffusion.plms_sample_loop_progressive


      image_display = Output()
      for i in range(args.n_batches):
          if args.animation_mode == 'None':
            display.clear_output(wait=True)
            batchBar = tqdm(range(args.n_batches), desc ="Batches")
            batchBar.n = i
            batchBar.refresh()
          print('')
          display.display(image_display)
          gc.collect()
          torch.cuda.empty_cache()
          cur_t = diffusion.num_timesteps - skip_steps - 1
          total_steps = cur_t

          if perlin_init:
              init = regen_perlin()

          if args.diffusion_sampling_mode == 'ddim':
              samples = sample_fn(
                  model,
                  (batch_size, 3, args.side_y, args.side_x),
                  clip_denoised=clip_denoised,
                  model_kwargs={},
                  cond_fn=cond_fn,
                  progress=True,
                  skip_timesteps=skip_steps,
                  init_image=init,
                  randomize_class=randomize_class,
                  eta=eta,
                  transformation_fn=symmetry_transformation_fn,
                  transformation_percent=args.transformation_percent
              )
          else:
              samples = sample_fn(
                  model,
                  (batch_size, 3, args.side_y, args.side_x),
                  clip_denoised=clip_denoised,
                  model_kwargs={},
                  cond_fn=cond_fn,
                  progress=True,
                  skip_timesteps=skip_steps,
                  init_image=init,
                  randomize_class=randomize_class,
                  order=2,
              )
          
#          textOverlay(text_overlays, img_filepath, frame_num)
          
          # with run_display:
          # display.clear_output(wait=True)
          for j, sample in enumerate(samples):    
            cur_t -= 1
            intermediateStep = False
            if args.steps_per_checkpoint is not None:
                if j % steps_per_checkpoint == 0 and j > 0:
                  intermediateStep = True
            elif j in args.intermediate_saves:
              intermediateStep = True
            with image_display:
              if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
                  for k, image in enumerate(sample['pred_xstart']):
                      # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                      current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
                      percent = math.ceil(j/total_steps*100)
                      if args.n_batches > 0:
                        #if intermediates are saved to the subfolder, don't append a step or percentage to the name
                        if cur_t == -1 and args.intermediates_in_subfolder is True:
                          save_num = f'{frame_num:04}' if animation_mode != "None" else i
                          filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
                        else:
                          #If we're working with percentages, append it
                          if args.steps_per_checkpoint is not None:
                            filename = f'{args.batch_name}({args.batchNum})_{i:04}-{percent:02}%.png'
                          # Or else, iIf we're working with specific steps, append those
                          else:
                            filename = f'{args.batch_name}({args.batchNum})_{i:04}-{j:03}.png'
                      image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                      #3dtorus
                      if audio_fx_3dtorus_enabled:
                        image = overlay3dtorus(audio_fx_blooming_data , image, frame_num)
                        #print('3dtorus')
                      #splat
                      if audio_fx_splat_enabled:
                        image = splat(audio_fx_blooming_data , image, frame_num, AudioSR)
                        #print('splat')
                      if audio_fx_polarwaveform_enabled:
                        image = polarOverlay(audio_fx_blooming_data , image, frame_num)
                        #print('polarwaveform')

                      ### apply text / VU
                      if frame_num in text_overlays:
                        image = textOverlay(text_overlays, image, frame_num)
                      if audio_fx_blooming_enabled:
                        image = bloomingOverlay(audio_fx_blooming_data , image, frame_num)
                        #print('blooming')
                      if j % args.display_rate == 0 or cur_t == -1:
                        image.save('progress.png')
                        display.clear_output(wait=True)
                        display.display(display.Image('progress.png'))
                      if args.steps_per_checkpoint is not None:
                        if j % args.steps_per_checkpoint == 0 and j > 0:
                          if args.intermediates_in_subfolder is True:
                            image.save(f'{partialFolder}/{filename}')
                          else:
                            image.save(f'{batchFolder}/{filename}')
                      else:
                        if j in args.intermediate_saves:
                          if args.intermediates_in_subfolder is True:
                            image.save(f'{partialFolder}/{filename}')
                          else:
                            image.save(f'{batchFolder}/{filename}')
                      if cur_t == -1:
                        if frame_num == 0:
                          save_settings()
                        if args.animation_mode != "None":
                          image.save('prevFrame.png')
                        image.save(f'{batchFolder}/{filename}')
                        if args.animation_mode == "3D":
                          # If turbo, save a blended image
                          if turbo_mode and frame_num > 0:
                            # Mix new image with prevFrameScaled
                            blend_factor = (1)/int(turbo_steps)
                            newFrame = cv2.imread('prevFrame.png') # This is already updated..
                            prev_frame_warped = cv2.imread('prevFrameScaled.png')
                            blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
                            cv2.imwrite(f'{batchFolder}/{filename}',blendedImage)
                          else:
                            image.save(f'{batchFolder}/{filename}')

                          if vr_mode:
                            generate_eye_views(TRANSLATION_SCALE, batchFolder, filename, frame_num, midas_model, midas_transform)

                        # if frame_num != args.max_frames-1:
                        #   display.clear_output()
          
          #plt.plot(np.array(loss_values), 'r')

def generate_eye_views(trans_scale,batchFolder,filename,frame_num,midas_model, midas_transform):
   for i in range(2):
      theta = vr_eye_angle * (math.pi/180)
      ray_origin = math.cos(theta) * vr_ipd / 2 * (-1.0 if i==0 else 1.0)
      ray_rotation = (theta if i==0 else -theta)
      translate_xyz = [-(ray_origin)*trans_scale, 0,0]
      rotate_xyz = [0, (ray_rotation), 0]
      rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
      transformed_image = dxf.transform_image_3d(f'{batchFolder}/{filename}', midas_model, midas_transform, DEVICE,
                                                      rot_mat, translate_xyz, args.near_plane, args.far_plane,
                                                      args.fov, padding_mode=args.padding_mode,
                                                      sampling_mode=args.sampling_mode, midas_weight=args.midas_weight,spherical=True)
      eye_file_path = batchFolder+f"/frame_{frame_num:04}" + ('_l' if i==0 else '_r')+'.png'
      transformed_image.save(eye_file_path)

def save_settings():
    setting_list = {
      'text_prompts': text_prompts,
      'image_prompts': image_prompts,
      'clip_guidance_scale': clip_guidance_scale,
      'tv_scale': tv_scale,
      'range_scale': range_scale,
      'sat_scale': sat_scale,
      # 'cutn': cutn,
      'cutn_batches': cutn_batches,
      'max_frames': max_frames,
      'interp_spline': interp_spline,
      # 'rotation_per_frame': rotation_per_frame,
      'init_image': init_image,
      'init_scale': init_scale,
      'skip_steps': skip_steps,
      # 'zoom_per_frame': zoom_per_frame,
      'frames_scale': frames_scale,
      'frames_skip_steps': frames_skip_steps,
      'perlin_init': perlin_init,
      'perlin_mode': perlin_mode,
      'skip_augs': skip_augs,
      'randomize_class': randomize_class,
      'clip_denoised': clip_denoised,
      'clamp_grad': clamp_grad,
      'clamp_max': clamp_max,
      'seed': seed,
      'fuzzy_prompt': fuzzy_prompt,
      'rand_mag': rand_mag,
      'eta': eta,
      'width': width_height[0],
      'height': width_height[1],
      'diffusion_model': diffusion_model,
      'use_secondary_model': use_secondary_model,
      'steps': steps,
      'diffusion_steps': diffusion_steps,
      'diffusion_sampling_mode': diffusion_sampling_mode,
      'ViTB32': ViTB32,
      'ViTB16': ViTB16,
      'ViTL14': ViTL14,
      'ViTL14_336px': ViTL14_336px,
      'RN101': RN101,
      'RN50': RN50,
      'RN50x4': RN50x4,
      'RN50x16': RN50x16,
      'RN50x64': RN50x64,
      'ViTB32_laion2b_e16': ViTB32_laion2b_e16,
      'ViTB32_laion400m_e31': ViTB32_laion400m_e31,
      'ViTB32_laion400m_32': ViTB32_laion400m_32,
      'ViTB32quickgelu_laion400m_e31': ViTB32quickgelu_laion400m_e31,
      'ViTB32quickgelu_laion400m_e32': ViTB32quickgelu_laion400m_e32,
      'ViTB16_laion400m_e31': ViTB16_laion400m_e31,
      'ViTB16_laion400m_e32': ViTB16_laion400m_e32,
      'RN50_yffcc15m': RN50_yffcc15m,
      'RN50_cc12m': RN50_cc12m,
      'RN50_quickgelu_yfcc15m': RN50_quickgelu_yfcc15m,
      'RN50_quickgelu_cc12m': RN50_quickgelu_cc12m,
      'RN101_yfcc15m': RN101_yfcc15m,
      'RN101_quickgelu_yfcc15m': RN101_quickgelu_yfcc15m,
      'cut_overview': str(cut_overview),
      'cut_innercut': str(cut_innercut),
      'cut_ic_pow': str(cut_ic_pow),
      'cut_icgray_p': str(cut_icgray_p),
      'key_frames': key_frames,
      'max_frames': max_frames,
      'angle': angle,
      'zoom': zoom,
      'translation_x': translation_x,
      'translation_y': translation_y,
      'translation_z': translation_z,
      'rotation_3d_x': rotation_3d_x,
      'rotation_3d_y': rotation_3d_y,
      'rotation_3d_z': rotation_3d_z,
      'midas_depth_model': midas_depth_model,
      'midas_weight': midas_weight,
      'near_plane': near_plane,
      'far_plane': far_plane,
      'fov': fov,
      'padding_mode': padding_mode,
      'sampling_mode': sampling_mode,
      'video_init_path':video_init_path,
      'extract_nth_frame':extract_nth_frame,
      'video_init_seed_continuity': video_init_seed_continuity,
      'turbo_mode':turbo_mode,
      'turbo_steps':turbo_steps,
      'turbo_preroll':turbo_preroll,
      'use_horizontal_symmetry':use_horizontal_symmetry,
      'use_vertical_symmetry':use_vertical_symmetry,
      'transformation_percent':transformation_percent,
      #video init settings
      'video_init_steps': video_init_steps,
      'video_init_clip_guidance_scale': video_init_clip_guidance_scale,
      'video_init_tv_scale': video_init_tv_scale,
      'video_init_range_scale': video_init_range_scale,
      'video_init_sat_scale': video_init_sat_scale,
      'video_init_cutn_batches': video_init_cutn_batches,
      'video_init_skip_steps': video_init_skip_steps,
      'video_init_frames_scale': video_init_frames_scale,
      'video_init_frames_skip_steps': video_init_frames_skip_steps,
      #warp settings
      'video_init_flow_warp':video_init_flow_warp,
      'video_init_flow_blend':video_init_flow_blend,
      'video_init_check_consistency':video_init_check_consistency,
      'video_init_blend_mode':video_init_blend_mode
    }
    # print('Settings:', setting_list)
    with open(f"{batchFolder}/{batch_name}({batchNum})_settings.txt", "w+", encoding="utf-8") as f:   #save settings
        json.dump(setting_list, f, ensure_ascii=False, indent=4)

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "MY3EGcMMLK_o"
# !! }}
#@title 1.5b Define necessary Audio functions
def panner(data,r=0.5):
  r = r * 2
  l = 2 - r
  y = (l * data, r * data)
  y = np.mean(y, axis=tuple(range(y.ndim - 1)))
  return y


def rangeLookup(freq_range):
  ranges={"Full":"mpout", "Sub Bass":"mpout_lp", "Bass":"mpout_bp1", "Lower Midrange":"mpout_bp2", "Midrange":"mpout_bp3", "Upper Midrange":"mpout_bp4", "Presence":"mpout_bp5", "Brilliance":"mpout_bp2", "Ceiling":"mpout_hp", "3StepLFO":"mpout_lfo","Sine":"mpout_sine","Saw":"mpout_saw","Square":"mpout_square" }
  if freq_range in ranges:
    return ranges[freq_range]
  else:
    return ranges["Full"]

def smoothing(data,kernel_size):
  kernel = np.ones(kernel_size) / kernel_size
  data_convolved = np.convolve(data, kernel, mode='same')
  return data_convolved

def normAndScalePositive(data,multiplier):
    #(x - min) / (max - min)
    max = np.max(data)
    min = np.min(data)
    data_scaled = np.array([(x - min) / (max - min) for x in data])
    return data_scaled*multiplier

def normAndScale(data,multiplier):
    #(x - min) / (max - min)
    max = np.max(data)
    min = np.min(data)
    data_scaled = np.array([(x - min) / (max - min) for x in data])
    return (2*data_scaled-1)*multiplier

def audioBandpassDataFetch(audio_data,smooth,kernel_size,multiplier):
    if smooth == True and kernel_size != '':
      audio_data = smoothing(audio_data,kernel_size)
    audio_data = normAndScale(audio_data,multiplier)

    return [  audio_data.frame_rate, audio_data ]


def threeStepsLFO(duration_seconds,multiplier):
    # 3 steps lfo
    a_param=0.5
    b_param=1.0
    c_param=0.0125
    three_step_lfo=a_param+pow((np.sin(duration_seconds*np.pi*2*c_param)*b_param),9)*0.5
    return normAndScale(three_step_lfo,multiplier);

def Basic_Square(duration_seconds,multiplier):
    ## Basic Square
    # a+Round(Frac(duration_seconds/(5-c*4.5)))*(b*2-1)-b+0.5
    a_param=0.5
    b_param=1.0
    c_param=0.025
    Square=a_param+np.round(np.modf(duration_seconds/(5-c_param*4.5))[0])*(b_param*2-1)-b_param+0.5
    return normAndScale(Square,multiplier);

def Basic_Sine(duration_seconds,multiplier):
    ## Basic Sine
    # a+Sin(duration_seconds*Pi*2*c)*(b-0.5) 
    a_param=0.5
    b_param=1.0
    c_param=0.0125
    Sine=a_param+np.sin(duration_seconds*np.pi*2*c_param)*(b_param-0.5) 
    return normAndScale(Sine,multiplier);

def Basic_Saw(duration_seconds,multiplier):
    ## Basic Saw
    # a+Frac(duration_seconds/(5-c*4.5))*(b*2-1)-b+0.5
    a_param=0.5
    b_param=1.0
    c_param=0.0125
    Saw=a_param+np.modf(duration_seconds/(5-c_param*4.5))[0]*(b_param*2-1)-b_param+0.5
    return normAndScale(Saw,multiplier);

#Quadratic (Quad), Cubic, Quartic, Quintic, Sine, Circular, Exponential, Elastic, Back, Bounce, Linear
#def Bounce_Ease_In_Out(duration_seconds,multiplier):
#    return BounceEaseInOut(start=multiplier, end=1, duration=duration_seconds)
def NormalizeData(data):
    data = data* (1/data.max())
    return data;

def MakeAudioData(data,title):
  strout=str()
  strcnt=0
  stroutData=np.array([], dtype=np.float64)
  while strcnt < max_iterations:
      cntStart = int(strcnt * blockSize)
      cntEnd = int((strcnt + 1) * blockSize - 1)
      #stroutData=np.append(stroutData, threeStepsLFO(strcnt), dtype=np.float64))

  strout.rstrip().rstrip(',')
  return strout

def plotGraph(data,title):
    plt.plot(data)
    plt.title(str(title))
    plt.show()
 

def MakeDataStr(data,title):
  plotGraph(data,title)
  strout=str()
  strcnt=0
  stroutData=np.array([], dtype=np.float64)
  while strcnt < max_iterations:
      strout = strout + str(strcnt)+ ": ("+str(data[strcnt])+"), "
      strcnt +=1
  strout.rstrip().rstrip(',')
  print(title," - ",strout)
  return strout

def percentage(part, whole):
  percentage = 100 * float(part)/float(whole)
  return str(percentage) + "%"

def floorToFrac(number,denominator):
  x = number * denominator
  x = np.floor(x)
  #print(number,x,denominator)
  x = x / denominator
  return x
## audio_cut_overview = "[12]*400+[4]*600"  
def MakeCutDataStr(dataIn,cutsTotal,splitamount):
  audio_cut_overview=str()
  audio_cut_innercut=str()
  stroutData=np.array([], dtype=np.float64)
  strinData=np.array([], dtype=np.float64)
  strcnt=0
  #print("splitamount:",splitamount)
  totalcuts=1000
  while totalcuts > len(dataIn):
    dataFlip = np.flip(dataIn)
    dataCon = np.concatenate((dataIn,dataFlip), axis=None)
    dataIn = dataCon[:totalcuts] 
  dataInParts=np.array_split(dataIn, splitamount)
  nearestNeighbor=((-totalcuts % splitamount) + totalcuts)
  #if totalcuts > len(dataInParts):
  #  dataInParts = np.reshape(np.concatenate(dataInParts,np.flip(dataInParts)),totalcuts)
  while strcnt < splitamount:
    #print(dataInParts[strcnt])
    splitValuePercent=floorToFrac(np.nanmean(dataInParts[strcnt]),cutsTotal)
    #print(strcnt,(cutsTotal*splitValuePercent),cutsTotal,splitValuePercent)
    splitValue=int(np.round(cutsTotal*splitValuePercent))
    audio_cut_overview=audio_cut_overview + '['+str(cutsTotal-splitValue)+str(']*')+str( int(nearestNeighbor/splitamount))+str('+')
    audio_cut_innercut=audio_cut_innercut + '['+str(splitValue)+str(']*')+str( int(nearestNeighbor/splitamount))+str('+')
    strinData=np.append(strinData,splitValue)
    stroutData=np.append(stroutData,(cutsTotal-splitValue))
    strcnt +=1
  audio_cut_innercut=audio_cut_innercut.rstrip().rstrip('+')
  audio_cut_overview=audio_cut_overview.rstrip().rstrip('+')
  plotGraph(stroutData,'audio_cut_overview')
  print('audio_cut_overview'," - ",audio_cut_overview)
  plotGraph(strinData,'audio_cut_innercut')
  print('audio_cut_innercut'," - ",audio_cut_innercut)
  return [audio_cut_overview,audio_cut_innercut]

  ## audio init image spectrogram funct
  
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled
def colorize(image, hue, saturation=1):
    """ Add color of the given hue to an RGB image.

    By default, set the saturation to 1 so that the colors pop!
    """
    from skimage import color
    #print(image)
    hsv = color.rgb2hsv(image)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return color.hsv2rgb(hsv)

def spectrogram_image(y, sr, out, hop_length, n_mels,time_steps):
    # use log-melspectrogram
    #fig, ax = plt.subplots(figsize=(1, 1))
    #fig, axes = plt.subplots(nrows=1, ncols=1)
    my_dpi=96
    borderRemovalScaleIncrease=30
    figHeight = (time_steps/my_dpi) + ((time_steps/my_dpi) * (borderRemovalScaleIncrease/100))
    figWidth = (n_mels/my_dpi) + ((n_mels/my_dpi) * (borderRemovalScaleIncrease/100))
    #print(figHeight,figWidth)
    #plt.figure(frameon=False ,figsize=( time_steps/my_dpi, n_mels/my_dpi), dpi=my_dpi)
    #plt.axis('off')
    #cmaps=['viridis', 'plasma', 'inferno', 'magma', 'cividis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper','PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic','twilight', 'twilight_shifted', 'hsv','Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c','flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar']
    cmaps=list(plt.colormaps())
    hue=int((np.sin(np.mean(y*sr) ))*len(cmaps))
    #print(len(cmaps))
    #print(cmaps[hue])
    cmap=plt.get_cmap(name=cmaps[hue])

    fig, ax =  plt.subplots(frameon=False ,figsize=(figHeight, figWidth), dpi=my_dpi)
    ax.axis("off")

    #ax.pcolormesh(cmap=cmap)
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels_dB = librosa.power_to_db(mels, ref=np.max)
    img = librosa.display.specshow(mels_dB, sr=sr, ax=ax)
    #mels = np.log(mels + 1e-9) # add small number to avoid log(0)
    img.set_cmap(cmap) 
    # min-max scale to fit inside 8-bit range
    #img = scale_minmax(mels, 0, 255).astype(np.uint8)
    #img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    #img = 255-img # invert. make black==more energy
    #quadmesh   https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap
    #colorize(img, hue, saturation=1)
    # save as PNG
    fig.savefig(out,dpi=my_dpi, bbox_inches='tight',pad_inches = 0)
    #skimage.io.imsave(out, img)


### Text Prompt Keyframe Recommender
 
#audio_fx_TPKR_type = "BeatDetection" #@param
# ["BeatDetection", "Superflux","PredominantLocalPulse","Backtracked"]
def plotBeatGraph(detType,data,title,fps):
  beatTime=np.array([], dtype=np.float64)

  if detType == 'BeatDetection' or detType == 'PredominantLocalPulse':
    onset_env = librosa.onset.onset_strength(y=data["y"], sr=data["sr"])
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=data["sr"])
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
    beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    fig, ax = plt.subplots(1)
    if detType == 'BeatDetection':
      times = librosa.times_like(onset_env, sr=data["sr"])
      ax.plot(times, librosa.util.normalize(onset_env),
         label='Onset strength')
      ax.vlines(times[beats], 0, 1, alpha=0.5, color='r',
           linestyle='--', label='Beats')
      ax.legend()
      ax.set(title='librosa.beat.beat_track')
      ax.label_outer()
      beatTime=times[beats]
    else: 
      times = librosa.times_like(pulse, sr=data["sr"])
      ax.plot(times, librosa.util.normalize(pulse),
         label='PLP')
      ax.vlines(times[beats_plp], 0, 1, alpha=0.5, color='r',
           linestyle='--', label='PLP Beats')
      ax.legend()
      ax.set(title='librosa.beat.plp')
      ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
      beatTime=times[beats_plp]
  elif detType == 'Superflux': 
    D = librosa.stft(data[y])
    D_harmonic, D_percussive = librosa.decompose.hpss(D)
    # Pre-compute a global reference power from the input spectrum
    rp = np.max(np.abs(D))
    n_fft = 1024
    hop_length = int(librosa.time_to_samples(1./200, sr=data["sr"]))
    lag = 2
    n_mels = 138
    fmin = 27.5
    fmax = 16000.
    max_size = 3
    S = librosa.feature.melspectrogram(y=data["y"], sr=data["sr"], n_fft=n_fft,
                                   hop_length=hop_length,
                                   fmin=fmin,
                                   fmax=fmax,
                                   n_mels=n_mels)
    odf_default = librosa.onset.onset_strength(y=data["y"], sr=data["sr"], hop_length=hop_length)
    onset_default = librosa.onset.onset_detect(y=data["y"], sr=data["sr"], hop_length=hop_length, units='time')
    odf_sf = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max),
                                      sr=data["sr"],
                                      hop_length=hop_length,
                                      lag=lag, max_size=max_size)
    onset_sf = librosa.onset.onset_detect(onset_envelope=odf_sf,
                                      sr=data["sr"],
                                      hop_length=hop_length,
                                      units='time')
    fig, ax = plt.subplots(nrows=1, sharex=True)
    frame_time = librosa.frames_to_time(np.arange(len(odf_default)),
                                    sr=data["sr"],
                                    hop_length=hop_length)
    ax.plot(frame_time, odf_sf, color='g', label='Superflux')
    ax.vlines(onset_sf, 0, odf_sf.max(), label='Onsets')
    ax.legend()
    ax.label_outer()
    beatTime=onset_sf
  elif detType == 'Backtracked':
    S = np.abs(librosa.stft(y=data["y"]))
    oenv = librosa.onset.onset_strength(y=data["y"], sr=data["sr"])
    times = librosa.times_like(oenv)
    # Detect events without backtracking
    onset_raw = librosa.onset.onset_detect(onset_envelope=oenv,
                                       backtrack=False)
    onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv)
    rms = librosa.feature.rms(S=S)
    onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, rms[0])
    fig, ax = plt.subplots()
    ax.plot(times, rms[0], label='RMS')
    beatTime=librosa.frames_to_time(onset_bt_rms)
    ax.vlines(beatTime, 0, rms.max(), label='Backtracked (RMS)', color='r')
    ax.legend()
  beatTime = [int(np.floor(bt * fps)) for bt in beatTime]
  return dict.fromkeys(beatTime)

### Prompt generation

def make_random_prompt(amount):
  for i in range(amount):
    
    prompt = []

    prompt_dict = {}


    # files to import
    prompt_dict["adjectives"] = open("/content/prompt_gen/adjectives.txt").read().splitlines()
    prompt_dict["animals"] = open("/content/prompt_gen/animals.txt").read().splitlines()
    prompt_dict["artists"] = open("/content/prompt_gen/artists.txt").read().splitlines()
    prompt_dict["colors"] = open("/content/prompt_gen/colors.txt").read().splitlines()
    prompt_dict["things"] = open("/content/prompt_gen/things.txt").read().splitlines()
    prompt_dict["shapes"] = open("/content/prompt_gen/shapes.txt").read().splitlines()
    prompt_dict["suffixes"] = open("/content/prompt_gen/styles.txt").read().splitlines()
    prompt_dict["locations"] = open("/content/prompt_gen/locations.txt").read().splitlines()
    

    adj_string = "" # for having multiple adjectives
    for adjective in range(n_adjectives):
        adj = random.choice(prompt_dict["adjectives"])
        prompt_dict["adjectives"].pop(prompt_dict["adjectives"].index(adj)) # to prevent repitition
                                      
        adj_string += adj + " "
    # prompt.append(adj_string) # uncomment for differnent separation formating

    clr_string = ""
    for color in range(n_colors):
        clr = random.choice(prompt_dict["colors"])
        prompt_dict["colors"].pop(prompt_dict["colors"].index(clr))
        clr_string += clr + " "
    # prompt.append(clr_string)

    if Animals: # not multiple since that would make things seem pretty weird
        animal = random.choice(prompt_dict["animals"]) + " "
        # prompt.append(animal)
    else:
        animal = ""

    if Shapes:
        shape = random.choice(prompt_dict["shapes"]) + " "
        # prompt.append(shape)
    else:
        shape = ""

    if Locations:
        location = random.choice(prompt_dict["locations"]) + " "
    else:
        location = ""


    thng_string = ""
    for thing in range(n_things):
        thng = random.choice(prompt_dict["things"])
        prompt_dict["things"].pop(prompt_dict["things"].index(thng))
        thng_string += thng + " "
    # prompt.append(thng_string)

    artst_string = ""
    if n_artists > 0:
        artst_string += "by "
    for artist in range(n_artists):
        artst = random.choice(prompt_dict["artists"])
        prompt_dict["artists"].pop(prompt_dict["artists"].index(artst))

        if artist < n_artists-1: # so that it doesn't add any unnecesary ands
            artst_string += artst + " and "
        else:
            artst_string += artst
    # prompt.append(artst_string)
    
    # adds sections together in the list
    prompt.append(f"{adj_string}{clr_string}{location}{shape}{animal}{thng_string}".strip())
    prompt.append(artst_string)
    
    for style in range(n_art_styles): # adds all the prompt suffixes separated by commas
        suffix = random.choice(prompt_dict["suffixes"])
        prompt_dict["suffixes"].pop(prompt_dict["suffixes"].index(suffix))
        prompt.append(suffix)

    # different formatting for different TTI. + a list comprehention to prevent useless commas at the end
    if VQGAN_prompt:
        promptList.append(" | ".join([i for i in prompt if i != "" and i != " "]))
    else:
        promptList.append(", ".join([i for i in prompt if i != "" and i != " "]))

### Overlay functions

### polarplot overlay
def Theta_funct(theta,a,b):        
    return a*np.sin(theta*b)   

#define a function in r
def R_cos(r,c,d):
    return c*np.cos(r*d)



## Blooming
# Alpha - feedback
# Hue
# Saturation
# Lightness
# BloomTint - adds a slight color tint to the bloom. 
# Iterations - depth of the effect.
# Dialation - spreads out the results.
# BloomLevel - mixing bloom with original image.
# Flicker - a random flux to the bloom (for neon sign, etc.).
# audio_fx_blooming_enabled = True #@param {type: "boolean" }
# audio_fx_blooming_preped=audio_fx_blooming_data = np.array([], dtype=np.float64)
# audio_fx_blooming_frequency = "Bass" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
# audio_fx_blooming_multiplier = 1 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
# audio_fx_blooming_soften = True #@param {type: "boolean" }
# audio_fx_blooming_kernel_size=20 #@param{type: 'integer'}

# Method to process the red band of the image
def normalizeRed(intensity):
    iI      = intensity
    minI    = 86
    maxI    = 230
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

# Method to process the green band of the image
def normalizeGreen(intensity):
    iI      = intensity
    minI    = 90
    maxI    = 225
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

# Method to process the blue band of the image
def normalizeBlue(intensity):
    iI      = intensity
    minI    = 100
    maxI    = 210
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

# Method to process the Alpha band of the image
def normalizeAlpha(intensity):
    iI      = intensity
    minI    = 100
    maxI    = 210
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

def normalizeBloom(intensity,j):
    iI      = intensity
    minI    = 0.2
    maxI    = 1
    minO    = 0
    maxO    = 1
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

def dilatationSize(intensity):
    iI      = intensity
    minI    = 0
    maxI    = 21
    minO    = 0
    maxO    = 21
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

def morphType(intensity):
    iI      = intensity
    minI    = 0
    maxI    = 2
    minO    = 0
    maxO    = 2
    iO      = int(np.round((iI-minI)*(((maxO-minO)/(maxI-minI))+minO)))
    return iO

ONE_SIXTEEN  = 1.0 / 16.0
FOUR_SIXTEEN = 4.0 / 16.0
SIX_SIXTEEN  = 6.0 / 16.0
ONE_HALF      = 1.0 / 2.0
ONE_FOURTH    = 1.0 / 4.0
ONE_EIGHTH    = 1.0 / 8.0
ONE_SIXTEENTH = 1.0 / 16.0

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE

def bloomingOverlay(audio_fx_blooming_data , image_bloom_in, j):
  if not (j % fps):
    return image_bloom_in
  # set arguments
  thresh_value = 245  # threshold to find white
  blur_value = 50     # bloom smoothness
  gain = 6            # bloom gain in intensity
  audData=audio_fx_blooming_data[j]
  audDataBlur=scale_minmax(audio_fx_blooming_data, min=0.0, max=21)
  blur_value=audDataBlur[j]
  audDataGain=scale_minmax(audio_fx_blooming_data, min=0.0, max=0.3)
  gain=audDataGain[j]
  #print(type(image_bloom_in))
  image_bloom= np.array(image_bloom_in) 
  image_bloom = cv2.cvtColor(image_bloom, cv2.COLOR_RGB2BGR)

  # convert image to hsv colorspace as floats
  hsv = cv2.cvtColor(image_bloom, cv2.COLOR_BGR2HSV).astype(np.float64)
  h, s, v = cv2.split(hsv)
  # Desire low saturation and high brightness for white
  # So invert saturation and multiply with brightness
  sv = ((255-s) * v / 255).clip(0,255).astype(np.uint8)
  # threshold
  thresh = cv2.threshold(sv, thresh_value, 255, cv2.THRESH_BINARY)[1]
  # blur and make 3 channels
  blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=blur_value, sigmaY=blur_value)
  blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
  #print(len(image_bloom.shape),len(blur.shape))
 
  # blend blur and image using gain on blur
  image_bloom = cv2.addWeighted(image_bloom, 1, blur, gain, 0)

  image_bloom = cv2.cvtColor(image_bloom, cv2.COLOR_BGR2RGB)
  im_pil = Image.fromarray(image_bloom)
  return  im_pil





def bloomingOverlay3(audio_fx_blooming_data , image_bloom, j):
  from PIL import ImageFilter
  with image_bloom.convert("RGBA") as base:
    BKERNEL = trunc(np.array(([ONE_SIXTEEN,
                               FOUR_SIXTEEN,
                               SIX_SIXTEEN,
                               FOUR_SIXTEEN,
                               ONE_SIXTEEN]), dtype=np.floating, copy=False), decs=5)
    audData=audio_fx_blooming_data[j]
    audDataScaled=scale_minmax(audio_fx_blooming_data, min=0.0, max=21)
    multiBands = base.split()
    # Apply point operations that does contrast stretching on each color band
    normalizedRedBand      = multiBands[0].point(normalizeRed)
    normalizedGreenBand    = multiBands[1].point(normalizeGreen)
    normalizedBlueBand     = multiBands[2].point(normalizeBlue)
    normalizedAlphaBand     = multiBands[3].point(normalizeAlpha)
    # Create a new image from the contrast stretched red, green and blue brands
    normalizedImage = Image.merge("RGBA", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand,normalizedAlphaBand))
    dilatation_size=int(float(dilatationSize(audDataScaled[j])))
    open_cv_normalizedImage  = cv2.cvtColor(np.array(normalizedImage), cv2.COLOR_RGBA2BGRA)
    #morph_shape(morphType(audio_fx_blooming_data,j))
    element = cv2.getStructuringElement(morph_shape(2), (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(open_cv_normalizedImage, element)
    open_cv_dilatation_dst  = cv2.cvtColor(dilatation_dst, cv2.COLOR_BGRA2RGBA)
    im_pil_dilatation_dst  = Image.fromarray(open_cv_dilatation_dst)
    blurred = im_pil_dilatation_dst.filter(ImageFilter.GaussianBlur(radius = audData))
    return blurred

def fig2rgb_array(fig):
    """adapted from: https://stackoverflow.com/questions/21939658/"""
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    #print("to verify, our resolution is: ",ncols,nrows)
    return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

def polarOverlay(y_block , imgdata , aj ):   
  return imgdata

def overlay3dtorus(y_block , imgdata , aj ):    
    # prepare some coordinates, and attach rgb values to each
    j=y_block[aj]
    r, theta, z = np.mgrid[0:1:11j, 0:np.pi*2:25j, -0.5:0.5:11j]
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    aud_vector = np.array(list(map(np.float, NormalizeData(y_block ))))
    aud_alpha=0.5+(aud_vector[aj]*0.5)
    rc, thetac, zc = midpoints(r), midpoints(theta), midpoints(z*j)

    # define a wobbly torus about [0.7, *, 0]
    sphere = (rc - 0.7)**2 + (zc + 0.2*np.cos(thetac*2))**2 < 0.2**2

    # combine the color components
    hsv = np.zeros(sphere.shape + (3,))
    hsv[..., 0] =  thetac / (np.pi*2) 
    hsv[..., 1] = rc
    hsv[..., 2] = zc + 0.5
    colors = mpl.colors.hsv_to_rgb(hsv)
    #color=colors.reshape(-1,4)
    # and plot everything
    #fig = plt.subplots(frameon=False,facecolor="None")  
    ofig = plt.figure(num=69,frameon=False,facecolor="None",clear=True)
    #ax = fig.add_axes(polar=True) 
    #ax = fig.add_subplot(polar=True) 
    oax = ofig.add_subplot(projection='3d')
    oax.axis("off")
    oax.grid(False)
    ofig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    #ofig.margins(0,0,0)
    oax.voxels(x, y, z, sphere,
              facecolors=colors,
              edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
              linewidth=0.5)

    polarOverlayImage = Image.fromarray(fig2rgb_array(ofig))
    polarOverlayImage = polarOverlayImage.convert('RGBA')
    polarOverlayImage = polarOverlayImage.resize(imgdata.size)
    #   Transparency
    newImage = []
    for item in polarOverlayImage.getdata():
     if item[:3] == (255, 255, 255):
       newImage.append((255, 255, 255, 0))
     else:
       newImage.append(item)
       

    #audDataAlpha= (np.floor( aud_vector*255))
    #aud_vector= (scale_minmax(audio_fx_polarwaveform_data, min=0 , max=128+128))
    # polarOverlayImage3 = polarOverlayImage.save("/content/polarOverlayImage.png")
    polarOverlayImage.putdata( newImage )
    #polarOverlayImage.putalpha(int(audDataAlpha[aj]))
    
    # polarOverlayImage4 = polarOverlayImage.save("/content/polarOverlayImage2.png")
    with imgdata.convert("RGBA") as base:
      out1 = Image.alpha_composite(base, polarOverlayImage)
      out = Image.blend(base, out1, alpha=aud_alpha)

    return out


def splat(y_block , imgdata , aj,sr):   
    # Fixing random state for reproducibility
    np.random.seed(int(float(y_block[aj] / sr)))
    splat_dpi=96
    # Compute areas and colors
    N =  int(  np.abs(float(y_block[aj]* sr)) /splat_dpi)
    r =  np.random.rand(N)
    theta = 2 * np.pi * np.random.rand(N)
    area =  400 * r**2 #(float(y_block[aj] / sr)*200)   * r**2
    colors = theta

    fig = plt.figure(frameon=False,facecolor="None")
    ax = fig.add_subplot(projection='polar')
    ax.axis("off")
    ax.grid(False)
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    # fig.margins(0,0)
    
    c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

    polarOverlayImage = Image.fromarray(fig2rgb_array(fig))
    polarOverlayImage = polarOverlayImage.convert('RGBA')
    polarOverlayImage = polarOverlayImage.resize(imgdata.size)
    #   Transparency
    newImage = []
    for item in polarOverlayImage.getdata():
     if item[:3] == (255, 255, 255):
       newImage.append((255, 255, 255, 0))
     else:
       newImage.append(item)
    #   print(item[:3])
    # polarOverlayImage3 = polarOverlayImage.save("/content/polarOverlayImage.png")
  
    polarOverlayImage.putdata(newImage)
    # polarOverlayImage4 = polarOverlayImage.save("/content/polarOverlayImage2.png")


    with imgdata.convert("RGBA") as base:
      out = Image.alpha_composite(base, polarOverlayImage)
    return out


# audio_fx_polarwaveform_enabled = True #@param {type: "boolean" }
# audio_fx_polarwaveform_preped=audio_fx_polarwaveform_data = np.array([], dtype=np.float64)
# audio_fx_polarwaveform_frequency = "Full" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
# audio_fx_polarwaveform_multiplier = 1 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
# audio_fx_polarwaveform_soften = True #@param {type: "boolean" }
# audio_fx_polarwaveform_kernel_size=20 #@param{type: 'integer'}
 

### text overlay
def textOverlay(textobj, img_filepath, frame_num):
  ## frame : [ "text to print:coordX,coordY:R,G,B,A:Font Size:Font Face" ]
  textstr, coord, img_RGBA, fontSize, fontFace =textobj[frame_num][0].split(':')
  coordX, coordY = coord.split(',')
  coordX, coordY = int(float(coordX)), int(float(coordY))
  img_R, img_G, img_B, img_A = img_RGBA.split(',')
  img_RGBA = (int(img_R), int(img_G), int(img_B), int(img_A))
  # get an image
  with img_filepath.convert("RGBA") as base:
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new("RGBA", base.size, (255, 255, 255, 0))
    # get a font
    fnt = ImageFont.truetype(fontFace, int(fontSize))
    # get a drawing context
    d = ImageDraw.Draw(txt)
    # draw text 
    d.text((coordX, coordY), textstr, font=fnt, fill=( img_RGBA ))
    out = Image.alpha_composite(base, txt)
    return out
### end text overlay

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "DefSecModel"
# !! }}
#@title 1.6 Define the secondary diffusion model

def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SecondaryDiffusionImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        self.timestep_embed = FourierFeatures(1, 16)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, c),
            ConvBlock(c, c),
            SkipBlock([
                nn.AvgPool2d(2),
                ConvBlock(c, c * 2),
                ConvBlock(c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),
                    ConvBlock(c * 2, c * 4),
                    ConvBlock(c * 4, c * 4),
                    SkipBlock([
                        nn.AvgPool2d(2),
                        ConvBlock(c * 4, c * 8),
                        ConvBlock(c * 8, c * 4),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ]),
                    ConvBlock(c * 8, c * 4),
                    ConvBlock(c * 4, c * 2),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ]),
                ConvBlock(c * 4, c * 2),
                ConvBlock(c * 2, c),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ]),
            ConvBlock(c * 2, c),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock([
                self.down,
                ConvBlock(cs[0], cs[1]),
                ConvBlock(cs[1], cs[1]),
                SkipBlock([
                    self.down,
                    ConvBlock(cs[1], cs[2]),
                    ConvBlock(cs[2], cs[2]),
                    SkipBlock([
                        self.down,
                        ConvBlock(cs[2], cs[3]),
                        ConvBlock(cs[3], cs[3]),
                        SkipBlock([
                            self.down,
                            ConvBlock(cs[3], cs[4]),
                            ConvBlock(cs[4], cs[4]),
                            SkipBlock([
                                self.down,
                                ConvBlock(cs[4], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[4]),
                                self.up,
                            ]),
                            ConvBlock(cs[4] * 2, cs[4]),
                            ConvBlock(cs[4], cs[3]),
                            self.up,
                        ]),
                        ConvBlock(cs[3] * 2, cs[3]),
                        ConvBlock(cs[3], cs[2]),
                        self.up,
                    ]),
                    ConvBlock(cs[2] * 2, cs[2]),
                    ConvBlock(cs[2], cs[1]),
                    self.up,
                ]),
                ConvBlock(cs[1] * 2, cs[1]),
                ConvBlock(cs[1], cs[0]),
                self.up,
            ]),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)

# %%
# !! {"metadata":{
# !!   "id": "DiffClipSetTop"
# !! }}
"""
# 2. Diffusion and CLIP model settings
"""

# %%
# !! {"metadata":{
# !!   "id": "ModelSettings"
# !! }}
#@markdown ####**Models Settings (note: For pixel art, the best is pixelartdiffusion_expanded):**
diffusion_model = "512x512_diffusion_uncond_finetune_008100" #@param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100", "portrait_generator_v001", "pixelartdiffusion_expanded", "pixel_art_diffusion_hard_256", "pixel_art_diffusion_soft_256", "pixelartdiffusion4k", "watercolordiffusion_2", "watercolordiffusion", "PulpSciFiDiffusion", "custom"]

use_secondary_model = True #@param {type: 'boolean'}
diffusion_sampling_mode = 'ddim' #@param ['plms','ddim']
#@markdown #####**Custom model:**
custom_path = '/content/drive/MyDrive/deep_learning/ddpm/ema_0.9999_058000.pt'#@param {type: 'string'}

#@markdown #####**CLIP settings:**
use_checkpoint = True #@param {type: 'boolean'}
ViTB32 = True #@param{type:"boolean"}
ViTB16 = True #@param{type:"boolean"}
ViTL14 = False #@param{type:"boolean"}
ViTL14_336px = False #@param{type:"boolean"}
RN101 = False #@param{type:"boolean"}
RN50 = True #@param{type:"boolean"}
RN50x4 = False #@param{type:"boolean"}
RN50x16 = False #@param{type:"boolean"}
RN50x64 = False #@param{type:"boolean"}

#@markdown #####**OpenCLIP settings:**
ViTB32_laion2b_e16 = False #@param{type:"boolean"}
ViTB32_laion400m_e31 = False #@param{type:"boolean"}
ViTB32_laion400m_32 = False #@param{type:"boolean"}
ViTB32quickgelu_laion400m_e31 = False #@param{type:"boolean"}
ViTB32quickgelu_laion400m_e32 = False #@param{type:"boolean"}
ViTB16_laion400m_e31 = False #@param{type:"boolean"}
ViTB16_laion400m_e32 = False #@param{type:"boolean"}
RN50_yffcc15m = False #@param{type:"boolean"}
RN50_cc12m = False #@param{type:"boolean"}
RN50_quickgelu_yfcc15m = False #@param{type:"boolean"}
RN50_quickgelu_cc12m = False #@param{type:"boolean"}
RN101_yfcc15m = False #@param{type:"boolean"}
RN101_quickgelu_yfcc15m = False #@param{type:"boolean"}

#@markdown Model checkpoints are always verified against the pinned SHA256 in
#@markdown diff_model_map before load -- torch.load(weights_only=False) below
#@markdown executes arbitrary code on unpickling, so an unverified checkpoint
#@markdown (corrupted, or served by a compromised/MITM'd mirror) is a real
#@markdown code-execution risk, not just a correctness one. This check is not
#@markdown optional.

diff_model_map = {
    '256x256_diffusion_uncond': { 'downloaded': False, 'sha': 'a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1', 'uri_list': ['https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', 'https://www.dropbox.com/s/9tqnqo930mpnpcn/256x256_diffusion_uncond.pt'] },
    '512x512_diffusion_uncond_finetune_008100': { 'downloaded': False, 'sha': '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648', 'uri_list': ['https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt', 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'] },
    'portrait_generator_v001': { 'downloaded': False, 'sha': 'b7e8c747af880d4480b6707006f1ace000b058dd0eac5bb13558ba3752d9b5b9', 'uri_list': ['https://huggingface.co/felipe3dartist/portrait_generator_v001/resolve/main/portrait_generator_v001_ema_0.9999_1MM.pt'] },
    'pixelartdiffusion_expanded': { 'downloaded': False, 'sha': 'a73b40556634034bf43b5a716b531b46fb1ab890634d854f5bcbbef56838739a', 'uri_list': ['https://huggingface.co/KaliYuga/PADexpanded/resolve/main/PADexpanded.pt'] },
    'pixel_art_diffusion_hard_256': { 'downloaded': False, 'sha': 'be4a9de943ec06eef32c65a1008c60ad017723a4d35dc13169c66bb322234161', 'uri_list': ['https://huggingface.co/KaliYuga/pixel_art_diffusion_hard_256/resolve/main/pixel_art_diffusion_hard_256.pt'] },
    'pixel_art_diffusion_soft_256': { 'downloaded': False, 'sha': 'd321590e46b679bf6def1f1914b47c89e762c76f19ab3e3392c8ca07c791039c', 'uri_list': ['https://huggingface.co/KaliYuga/pixel_art_diffusion_soft_256/resolve/main/pixel_art_diffusion_soft_256.pt'] },
    'pixelartdiffusion4k': { 'downloaded': False, 'sha': 'a1ba4f13f6dabb72b1064f15d8ae504d98d6192ad343572cc416deda7cccac30', 'uri_list': ['https://huggingface.co/KaliYuga/pixelartdiffusion4k/resolve/main/pixelartdiffusion4k.pt'] },
    'watercolordiffusion_2': { 'downloaded': False, 'sha': '49c281b6092c61c49b0f1f8da93af9b94be7e0c20c71e662e2aa26fee0e4b1a9', 'uri_list': ['https://huggingface.co/KaliYuga/watercolordiffusion_2/resolve/main/watercolordiffusion_2.pt'] },
    'watercolordiffusion': { 'downloaded': False, 'sha': 'a3e6522f0c8f278f90788298d66383b11ac763dd5e0d62f8252c962c23950bd6', 'uri_list': ['https://huggingface.co/KaliYuga/watercolordiffusion/resolve/main/watercolordiffusion.pt'] },
    'PulpSciFiDiffusion': { 'downloaded': False, 'sha': 'b79e62613b9f50b8a3173e5f61f0320c7dbb16efad42a92ec94d014f6e17337f', 'uri_list': ['https://huggingface.co/KaliYuga/PulpSciFiDiffusion/resolve/main/PulpSciFiDiffusion.pt'] },
    'secondary': { 'downloaded': False, 'sha': '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a', 'uri_list': ['https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth', 'https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth', 'https://ipfs.pollinations.ai/ipfs/bafybeibaawhhk7fhyhvmm7x24zwwkeuocuizbqbcg5nqx64jq42j75rdiy/secondary_model_imagenet_2.pth'] },
}

kaliyuga_pixel_art_model_names = ['pixelartdiffusion_expanded', 'pixel_art_diffusion_hard_256', 'pixel_art_diffusion_soft_256', 'pixelartdiffusion4k', 'PulpSciFiDiffusion']
kaliyuga_watercolor_model_names = ['watercolordiffusion', 'watercolordiffusion_2']
kaliyuga_pulpscifi_model_names = ['PulpSciFiDiffusion']
diffusion_models_256x256_list = ['256x256_diffusion_uncond'] + kaliyuga_pixel_art_model_names + kaliyuga_watercolor_model_names + kaliyuga_pulpscifi_model_names

from urllib.parse import urlparse

def get_model_filename(diffusion_model_name):
    model_uri = diff_model_map[diffusion_model_name]['uri_list'][0]
    model_filename = os.path.basename(urlparse(model_uri).path)
    return model_filename


def _sha256_matches(path, expected_sha):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest() == expected_sha


def download_model(diffusion_model_name, uri_index=0):
    if diffusion_model_name != 'custom':
        model_filename = get_model_filename(diffusion_model_name)
        model_local_path = os.path.join(model_path, model_filename)
        expected_sha = diff_model_map[diffusion_model_name]['sha']

        if os.path.exists(model_local_path):
            print(f'Checking {diffusion_model_name} file against its pinned SHA256')
            if _sha256_matches(model_local_path, expected_sha):
                print(f'{diffusion_model_name} SHA matches')
                diff_model_map[diffusion_model_name]['downloaded'] = True
            else:
                print(f"{diffusion_model_name} SHA doesn't match -- deleting and redownloading.")
                os.remove(model_local_path)

        if not diff_model_map[diffusion_model_name]['downloaded']:
            for model_uri in diff_model_map[diffusion_model_name]['uri_list']:
                wget(model_uri, model_path)
                if not os.path.exists(model_local_path):
                    print(f'{diffusion_model_name} model download from {model_uri} failed. Will try any fallback uri.')
                    continue
                if _sha256_matches(model_local_path, expected_sha):
                    diff_model_map[diffusion_model_name]['downloaded'] = True
                    return
                print(f'{diffusion_model_name} downloaded from {model_uri} does not match the pinned '
                      f'SHA256 -- deleting and trying the next mirror.')
                os.remove(model_local_path)
            raise RuntimeError(
                f'{diffusion_model_name} download failed: no mirror in uri_list produced a file '
                f'matching the pinned SHA256. Refusing to load an unverified checkpoint.'
            )


# Download the diffusion model(s)
download_model(diffusion_model)
if use_secondary_model:
    download_model('secondary')

model_config = model_and_diffusion_defaults()
if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
        'rescale_timesteps': True,
        'timestep_respacing': 250, #No need to edit this, it is taken care of later.
        'image_size': 512,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': use_checkpoint,
        'use_fp16': not useCPU,
        'use_scale_shift_norm': True,
    })
elif diffusion_model == '256x256_diffusion_uncond':
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000, #No need to edit this, it is taken care of later.
        'rescale_timesteps': True,
        'timestep_respacing': 250, #No need to edit this, it is taken care of later.
        'image_size': 256,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': use_checkpoint,
        'use_fp16': not useCPU,
        'use_scale_shift_norm': True,
    })
elif diffusion_model == 'portrait_generator_v001':
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'image_size': 512,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 128,
        'num_heads': 4,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': use_checkpoint,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })
else:  # E.g. A model finetuned by KaliYuga
    model_config.update({
          'attention_resolutions': '16',
          'class_cond': False,
          'diffusion_steps': 1000,
          'rescale_timesteps': True,
          'timestep_respacing': 'ddim100',
          'image_size': 256,
          'learn_sigma': True,
          'noise_schedule': 'linear',
          'num_channels': 128,
          'num_heads': 1,
          'num_res_blocks': 2,
          'use_checkpoint': use_checkpoint,
          'use_fp16': True,
          'use_scale_shift_norm': False,
      })

model_default = model_config['image_size']

if use_secondary_model:
    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(torch.load(f'{model_path}/secondary_model_imagenet_2.pth', map_location='cpu', weights_only=False))
    secondary_model.eval().requires_grad_(False).to(device)

clip_models = []
if ViTB32: clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device))
if ViTB16: clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device))
if ViTL14: clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device))
if ViTL14_336px: clip_models.append(clip.load('ViT-L/14@336px', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50: clip_models.append(clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x4: clip_models.append(clip.load('RN50x4', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x16: clip_models.append(clip.load('RN50x16', jit=False)[0].eval().requires_grad_(False).to(device))
if RN50x64: clip_models.append(clip.load('RN50x64', jit=False)[0].eval().requires_grad_(False).to(device))
if RN101: clip_models.append(clip.load('RN101', jit=False)[0].eval().requires_grad_(False).to(device))
if ViTB32_laion2b_e16: clip_models.append(open_clip.create_model('ViT-B-32', pretrained='laion2b_e16').eval().requires_grad_(False).to(device))
if ViTB32_laion400m_e31: clip_models.append(open_clip.create_model('ViT-B-32', pretrained='laion400m_e31').eval().requires_grad_(False).to(device))
if ViTB32_laion400m_32: clip_models.append(open_clip.create_model('ViT-B-32', pretrained='laion400m_e32').eval().requires_grad_(False).to(device))
if ViTB32quickgelu_laion400m_e31: clip_models.append(open_clip.create_model('ViT-B-32-quickgelu', pretrained='laion400m_e31').eval().requires_grad_(False).to(device))
if ViTB32quickgelu_laion400m_e32: clip_models.append(open_clip.create_model('ViT-B-32-quickgelu', pretrained='laion400m_e32').eval().requires_grad_(False).to(device))
if ViTB16_laion400m_e31: clip_models.append(open_clip.create_model('ViT-B-16', pretrained='laion400m_e31').eval().requires_grad_(False).to(device))
if ViTB16_laion400m_e32: clip_models.append(open_clip.create_model('ViT-B-16', pretrained='laion400m_e32').eval().requires_grad_(False).to(device))
if RN50_yffcc15m: clip_models.append(open_clip.create_model('RN50', pretrained='yfcc15m').eval().requires_grad_(False).to(device))
if RN50_cc12m: clip_models.append(open_clip.create_model('RN50', pretrained='cc12m').eval().requires_grad_(False).to(device))
if RN50_quickgelu_yfcc15m: clip_models.append(open_clip.create_model('RN50-quickgelu', pretrained='yfcc15m').eval().requires_grad_(False).to(device))
if RN50_quickgelu_cc12m: clip_models.append(open_clip.create_model('RN50-quickgelu', pretrained='cc12m').eval().requires_grad_(False).to(device))
if RN101_yfcc15m: clip_models.append(open_clip.create_model('RN101', pretrained='yfcc15m').eval().requires_grad_(False).to(device))
if RN101_quickgelu_yfcc15m: clip_models.append(open_clip.create_model('RN101-quickgelu', pretrained='yfcc15m').eval().requires_grad_(False).to(device))

normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)

# %%
# !! {"metadata":{
# !!   "id": "CustModelTop"
# !! }}
"""
# Custom model settings 
Modify in accordance with your training settings and run the cell
"""

# %%
# !! {"metadata":{
# !!   "id": "CustModel"
# !! }}
#@markdown ####**Custom Model Settings:**
if diffusion_model == 'custom':
  model_config.update({
          'attention_resolutions': '16',
          'class_cond': False,
          'diffusion_steps': 1000,
          'rescale_timesteps': True,
          'timestep_respacing': 'ddim100',
          'image_size': 256,
          'learn_sigma': True,
          'noise_schedule': 'linear',
          'num_channels': 128,
          'num_heads': 1,
          'num_res_blocks': 2,
          'use_checkpoint': use_checkpoint,
          'use_fp16': True,
          'use_scale_shift_norm': False,
      })

# %%
# !! {"metadata":{
# !!   "id": "SettingsTop"
# !! }}
"""
# 3. Settings
"""

# %%
# !! {"metadata":{
# !!   "id": "BasicSettings"
# !! }}

#@title generate spectrogram for init image
from os.path import exists as file_exists
out = f'{root_path}/input/init.png'
audio_filename = r'C:\audiorider\audio_samples\Gui - Ocean Eyes Liquid Edit.wav'
createPath(os.path.dirname(out))
if file_exists(out):
  print("init image exists already ", out)
else:
  # settings
  hop_length = 512 # number of samples per time-step in spectrogram
  n_mels = 720 # number of bins in spectrogram. Height of image
  time_steps = 1280 # number of time-steps. Width of image

  # load audio. Using example from librosa 
  #y, sr = librosa.load(audio_filename, offset=1.0, duration=10.0, sr=22050)
  y, sr = librosa.load(audio_filename)
   
  # extract a fixed length window
  start_sample = 0 # starting at beginning
  length_samples = time_steps*hop_length
  window = y[start_sample:start_sample+length_samples]
    
  # convert to PNG
  spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels,time_steps=time_steps)
  print('wrote file ', out) 
  

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "Dp7Z4LmSNyM_"
# !! }}
#@title get resolution from init image and scale silder
import os
import cv2
import math
from decimal import Decimal


img2 = cv2.imread(out)
  
# fetching the dimensions
nXPixels = img2.shape[1]
nYPixels = img2.shape[0]
  
#nXPixels, nYPixels = Image.
#@markdown Get init image aspect ratio, and create resolution based off of the megapixel slider.
nNewPixelResolution=230000 #@param {type:"slider", min:10000, max:1000000, step:10000}

nOldPixelResolution=nXPixels*nYPixels
conv = Decimal(math.sqrt( Decimal(nNewPixelResolution)/Decimal(nOldPixelResolution)))
width=  round(nXPixels*conv)
height= round(nYPixels*conv)
print("resolution you should use: ("+str(width)+" "+str(height)+")")
print("setting env vars 'PROMPT_W' and 'PROMPT_H' ")
os.environ['PROMPT_W'] =str(width)
os.environ['PROMPT_H'] =str(height)
# The original used `!export | grep ...` to echo these back -- a bash builtin,
# not available via Windows cmd.exe. Plain Python works on every platform.
print("PROMPT_W =", os.environ['PROMPT_W'])
print("PROMPT_H =", os.environ['PROMPT_H'])

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "BasicSettings"
# !! }}
#@markdown ####**Basic Settings:**
# batch_basename was referenced here but never defined anywhere in the
# original notebook -- a real bug, not a Windows/Colab difference.
# Deriving a sensible default from the audio file already in scope.
batch_basename = os.path.splitext(os.path.basename(audio_filename))[0]
batch_name = f'{batch_basename}' #@param{type: 'string'}
steps = 500 #@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
width_height_for_512x512_models = [int(os.environ['PROMPT_W']), int(os.environ['PROMPT_H'])] #@param{type: 'raw'}
clip_guidance_scale = 5000 #@param{type: 'number'}
tv_scale =  0#@param{type: 'number'}
range_scale = 150#@param{type: 'number'}
sat_scale = 0#@param{type: 'number'}
cutn_batches = 4#@param{type: 'number'}
skip_augs = False#@param{type: 'boolean'}

#@markdown ####**Image dimensions to be used for 256x256 models (e.g. pixelart models):**
width_height_for_256x256_models = [512, 448] #@param{type: 'raw'}

#@markdown ####**Video Init Basic Settings:**
video_init_steps = 100 #@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
video_init_clip_guidance_scale = 1000 #@param{type: 'number'}
video_init_tv_scale = 0.1#@param{type: 'number'}
video_init_range_scale = 150#@param{type: 'number'}
video_init_sat_scale = 300#@param{type: 'number'}
video_init_cutn_batches = 4#@param{type: 'number'}
video_init_skip_steps = 50 #@param{type: 'integer'}

#@markdown ---

#@markdown ####**Init Image Settings:**
init_image = out #@param{type: 'string'}
init_scale =  1000#@param{type: 'integer'}
skip_steps_percent=0.65 #@param {type:"slider", min:0, max:0.9, step:0.05}
skip_steps = int(steps*skip_steps_percent) #@ param{type: 'integer'}
#@markdown *Make sure you set skip_steps to ~50% of your steps if you want to use an init image.*

width_height = width_height_for_256x256_models if diffusion_model in diffusion_models_256x256_list else width_height_for_512x512_models

#Get corrected sizes
side_x = (width_height[0]//64)*64;
side_y = (width_height[1]//64)*64;
if side_x != width_height[0] or side_y != width_height[1]:
    print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.')

#Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000//steps)*steps if steps < 1000 else steps
model_config.update({
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
})

#Make folder for batch
batchFolder = f'{outDirPath}/{batch_name}'
createPath(batchFolder)

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "KOMeoHDZ2UIh"
# !! }}
#@title # ___Audio Settings___
init_audio=audio_filename #@param{type: 'string'}
#@markdown $\color{red}{\text{Sub Bass}}$ 0-60Hz </br>
#@markdown $\color{orange}{\text{Bass}}$ 60-250Hz </br>
#@markdown $\color{yellow}{\text{Lower Midrange}}$ 250-500Hz</br>
#@markdown $\color{limegreen}{\text{Midrange}}$ 500-2000Hz </br>
#@markdown $\color{skyblue}{\text{Upper Midrange}}$ 2000-4000Hz </br>
#@markdown $\color{teal}{\text{Presence}}$ 4000-6000Hz</br>
#@markdown $\color{lavender}{\text{Brilliance / Air }}$ 6000-20,000Hz</br>
fps=12 #@param{type: 'integer'}
decimalsPlaces=6#@param{type: 'integer'}

bandsWillMakeHerDanceDesc={"band0":"Floor", "band1":"Sub Bass", "band2":"Bass", "band3":"Lower Midrange", "band4":"Midrange", "band5":"Upper Midrange", "band6":"Presence", "band7":"Brilliance", "band8":"Ceiling" }
bandsWillMakeHerDance2={"band0":0, "band1":60, "band2":250, "band3":500, "band4":2000, "band5":4000, "band6":6000, "band7":17000, "band8":20000 }

audio_data = AudioSegment.from_mp3(init_audio)    
sampleLen = audio_data.duration_seconds 

a = AudioSegment.from_mp3(init_audio)
AudioSR=a.frame_rate
a_lp=a.low_pass_filter(bandsWillMakeHerDance2["band1"])
a_hp=a.high_pass_filter(bandsWillMakeHerDance2["band7"])
    
a_bp1=a.band_pass_filter(bandsWillMakeHerDance2["band1"],bandsWillMakeHerDance2["band2"])
a_bp2=a.band_pass_filter(bandsWillMakeHerDance2["band2"],bandsWillMakeHerDance2["band3"])
a_bp3=a.band_pass_filter(bandsWillMakeHerDance2["band3"],bandsWillMakeHerDance2["band4"])
a_bp4=a.band_pass_filter(bandsWillMakeHerDance2["band4"],bandsWillMakeHerDance2["band5"])
a_bp5=a.band_pass_filter(bandsWillMakeHerDance2["band5"],bandsWillMakeHerDance2["band6"])
a_bp6=a.band_pass_filter(bandsWillMakeHerDance2["band6"],bandsWillMakeHerDance2["band7"])

y     =  np.array(a.get_array_of_samples())     
y_lp  =  np.array(a_lp.get_array_of_samples())  
y_hp  =  np.array(a_hp.get_array_of_samples())  
y_bp1 =  np.array(a_bp1.get_array_of_samples()) 
y_bp2 =  np.array(a_bp2.get_array_of_samples()) 
y_bp3 =  np.array(a_bp3.get_array_of_samples()) 
y_bp4 =  np.array(a_bp4.get_array_of_samples()) 
y_bp5 =  np.array(a_bp5.get_array_of_samples()) 
y_bp6 =  np.array(a_bp6.get_array_of_samples()) 
sampleLen = a.duration_seconds 

y_sine=Basic_Sine(y,1)
y_square=Basic_Square(y,1)
y_saw=Basic_Saw(y,1)
y_lfo=threeStepsLFO(y,1)

if a.channels == 2:
    y = y.reshape((-1, 2))
    y_lp = y_lp.reshape((-1, 2))
    y_hp = y_hp.reshape((-1, 2))
    y_bp1 = y_bp1.reshape((-1, 2))
    y_bp2 = y_bp2.reshape((-1, 2))
    y_bp3 = y_bp3.reshape((-1, 2))
    y_bp4 = y_bp4.reshape((-1, 2))
    y_bp5 = y_bp5.reshape((-1, 2))
    y_bp6 = y_bp6.reshape((-1, 2))

music_prompt2 = [  a.frame_rate, NormalizeData(y) ]
mp_lp = [  a_lp.frame_rate, NormalizeData(y_lp) ]
mp_hp = [  a_hp.frame_rate, NormalizeData(y_hp) ]
mp_bp1 = [  a_bp1.frame_rate, NormalizeData(y_bp1) ]
mp_bp2 = [  a_bp2.frame_rate, NormalizeData(y_bp2) ]
mp_bp3 = [  a_bp3.frame_rate, NormalizeData(y_bp3) ]
mp_bp4 = [  a_bp4.frame_rate, NormalizeData(y_bp4) ]
mp_bp5 = [  a_bp5.frame_rate, NormalizeData(y_bp5) ]
mp_bp6 = [  a_bp6.frame_rate, NormalizeData(y_bp6) ]

mp_sine = [  a.frame_rate,  y_sine  ]
mp_square = [  a.frame_rate,  y_square  ]
mp_saw = [  a.frame_rate,  y_saw  ]
mp_lfo = [  a.frame_rate,  y_lfo  ]

###
max_iterations=int(sampleLen)*int(fps)
print("using mp3 length for iterations: "+str(max_iterations))
blockSize = np.floor(len(y) / max_iterations)
cnt=0
mpout_sine = mpout_square = mpout_saw = mpout_lfo = mpout =  mpout_lp = mpout_hp = mpout_bp1 = mpout_bp2 = mpout_bp3 = mpout_bp4 = mpout_bp5 = mpout_bp6 = np.array([], dtype=np.float64)
while cnt < max_iterations:
    cntStart = int(cnt * blockSize)
    cntEnd = int((cnt + 1) * blockSize - 1)

    # music_prompt2 = [  a.frame_rate, NormalizeData(y) ]
    mpout=np.append(mpout,  (np.mean(music_prompt2[1][cntStart:cntEnd], dtype=np.float64)))
    # mp_lp = [  a_lp.frame_rate, NormalizeData(y_lp) ]
    mpout_lp=np.append(mpout_lp,  (np.mean(mp_lp[1][cntStart:cntEnd], dtype=np.float64)))
    # mp_hp = [  a_hp.frame_rate, NormalizeData(y_hp) ]
    mpout_hp=np.append(mpout_hp,  (np.mean(mp_hp[1][cntStart:cntEnd], dtype=np.float64)))
    # mp_bp1 = [  a_bp1.frame_rate, NormalizeData(y_bp1) ]
    mpout_bp1=np.append(mpout_bp1, (np.mean(mp_bp1[1][cntStart:cntEnd], dtype=np.float64)))
    # mp_bp2 = [  a_bp2.frame_rate, NormalizeData(y_bp2) ]
    mpout_bp2=np.append(mpout_bp2, (np.mean(mp_bp2[1][cntStart:cntEnd], dtype=np.float64)))
    # mp_bp3 = [  a_bp3.frame_rate, NormalizeData(y_bp3) ]
    mpout_bp3=np.append(mpout_bp3,  (np.mean(mp_bp3[1][cntStart:cntEnd], dtype=np.float64)))
    # mp_bp4 = [  a_bp4.frame_rate, NormalizeData(y_bp4) ]
    mpout_bp4=np.append(mpout_bp4, (np.mean(mp_bp4[1][cntStart:cntEnd], dtype=np.float64)))
    # mp_bp5 = [  a_bp5.frame_rate, NormalizeData(y_bp5) ]
    mpout_bp5=np.append(mpout_bp5, (np.mean(mp_bp5[1][cntStart:cntEnd], dtype=np.float64)))
    # mp_bp6 = [  a_bp6.frame_rate, NormalizeData(y_bp6) ]
    mpout_bp6=np.append(mpout_bp6, (np.mean(mp_bp6[1][cntStart:cntEnd], dtype=np.float64))) 
    mpout_sine=np.append(mpout_sine, (np.mean(mp_sine[1][cntStart:cntEnd], dtype=np.float64))) 
    mpout_square=np.append(mpout_square, (np.mean(mp_square[1][cntStart:cntEnd], dtype=np.float64))) 
    mpout_saw=np.append(mpout_saw, (np.mean(mp_saw[1][cntStart:cntEnd], dtype=np.float64))) 
    mpout_lfo=np.append(mpout_lfo, (np.mean(mp_lfo[1][cntStart:cntEnd], dtype=np.float64))) 
    cnt = cnt + 1

#@markdown kernel_size used for smoothing audio data. larger kernel size is more smooth output

#@markdown # ___2d Settings___ #

#@markdown audio_angle: angle: (0|-3 to 3)  
audio_angle_preped=audio_angle_data=np.array([], dtype=np.float64)
audio_angle_frequency = "Sub Bass" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_angle_multiplier = 3 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_angle_soften = True #@param {type: "boolean" }
audio_angle_kernel_size=20 #@param{type: 'integer'}
audio_angle_data =  locals()[rangeLookup(audio_angle_frequency)]
if audio_angle_soften == True and audio_angle_kernel_size != '':
  audio_angle_data = smoothing(audio_angle_data,audio_angle_kernel_size)
audio_angle_data = normAndScale(audio_angle_data,audio_angle_multiplier)
audio_angle=str()

#@markdown audio_zoom: zoom: (2D only) (1.10|0.8 - 1.25)  
audio_zoom_preped=audio_zoom_data=np.array([], dtype=np.float64)
audio_zoom_frequency = "Bass" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_zoom_multiplier = 0.5 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_zoom_soften = True #@param {type: "boolean" }
audio_zoom_kernel_size=20 #@param{type: 'integer'}
audio_zoom_data =  locals()[rangeLookup(audio_zoom_frequency)]
if audio_zoom_soften == True and audio_zoom_kernel_size != '':
  audio_zoom_data = smoothing(audio_zoom_data,audio_zoom_kernel_size)
audio_zoom_data = normAndScale(audio_zoom_data,audio_zoom_multiplier)
audio_zoom=str()
#@markdown # ___3d Settings___ #
#@markdown ##### rotation_3d_x, rotation_3d_y, rotation_3d_z : (0|-0.05 to 0.05)
#@markdown ##### translation_x, translation_y, translation_z (in 3D mode): (0|-10 to 10)

#@markdown ___audio_translation_x___: X is left/right; positive translation_x shifts the camera to the right
audio_translation_x_preped=audio_translation_x_data=np.array([], dtype=np.float64)
audio_translation_x_frequency = "Midrange" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_translation_x_multiplier = 4 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_translation_x_soften = True #@param {type: "boolean" }
audio_translation_x_kernel_size=216 #@param{type: 'integer'}
audio_translation_x_data =  locals()[rangeLookup(audio_translation_x_frequency)]
if audio_translation_x_soften == True and audio_translation_x_kernel_size != '':
  audio_translation_x_data = smoothing(audio_translation_x_data,audio_translation_x_kernel_size)
audio_translation_x_data = normAndScale(audio_translation_x_data,audio_translation_x_multiplier)
audio_translation_x=str()
#@markdown ___audio_translation_y___: y is up/down; positive translation_y shifts the camera upward
audio_translation_y_preped=audio_translation_y_data =np.array([], dtype=np.float64)
audio_translation_y_frequency = "Upper Midrange" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_translation_y_multiplier = 4 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_translation_y_soften = True #@param {type: "boolean" }
audio_translation_y_kernel_size=216 #@param{type: 'integer'}
audio_translation_y_data =  locals()[rangeLookup(audio_translation_y_frequency)]
if audio_translation_y_soften == True and audio_translation_y_kernel_size != '':
  audio_translation_y_data = smoothing(audio_translation_y_data,audio_translation_y_kernel_size)
audio_translation_y_data = normAndScale(audio_translation_y_data,audio_translation_y_multiplier)
audio_translation_y=str()
#@markdown ___audio_translation_z___: z is forward/backwards (zooming); positive translation_z shifts the camera forward 
audio_translation_z_preped=audio_translation_z_data =np.array([], dtype=np.float64)
audio_translation_z_frequency = "Sine" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_translation_z_multiplier = 3 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_translation_z_soften = True #@param {type: "boolean" }
audio_translation_z_kernel_size=216 #@param{type: 'integer'}
audio_translation_z_modulation = True #@param {type: "boolean" }
audio_translation_z_modfreq = "Bass" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_translation_z_modmultiplier = 2 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_translation_z_modsoften = True #@param {type: "boolean" }
audio_translation_z_modkernel_size=216 #@param{type: 'integer'}
if audio_translation_z_modulation == True:
  audio_translation_z_modfreq_data = np.array([], dtype=np.float64)
  audio_translation_z_modfreq_data = locals()[rangeLookup(audio_translation_z_modfreq)]
  if audio_translation_z_modsoften == True and audio_translation_z_modkernel_size != '':
    audio_translation_z_modfreq_data = smoothing(audio_translation_z_modfreq_data,audio_translation_z_modkernel_size)
  audio_translation_z_modfreq_data = normAndScale(audio_translation_z_modfreq_data,audio_translation_z_modmultiplier)

audio_translation_z_data =  locals()[rangeLookup(audio_translation_z_frequency)]
if audio_translation_z_soften == True and audio_translation_z_kernel_size != '':
  audio_translation_z_data = smoothing(audio_translation_z_data,audio_translation_z_kernel_size)
audio_translation_z_data = normAndScale(audio_translation_z_data,audio_translation_z_multiplier)
audio_translation_z=str()

if audio_translation_z_modulation == True:
  audio_translation_z_data =     audio_translation_z_data* (audio_translation_z_modfreq_data+1 )

#@markdown ___audio_rotation_3d_x___: Positive rotation_3d_x pitches the camera upward.
audio_rotation_3d_x_preped=audio_rotation_3d_x_data=np.array([], dtype=np.float64)
audio_rotation_3d_x_frequency = "Midrange" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_rotation_3d_x_multiplier = 0.25 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_rotation_3d_x_soften = True #@param {type: "boolean" }
audio_rotation_3d_x_kernel_size=216 #@param{type: 'integer'}
audio_rotation_3d_x_data =  locals()[rangeLookup(audio_rotation_3d_x_frequency)]
if audio_rotation_3d_x_soften == True and audio_rotation_3d_x_kernel_size != '':
  audio_rotation_3d_x_data = smoothing(audio_rotation_3d_x_data,audio_rotation_3d_x_kernel_size)
audio_rotation_3d_x_data = normAndScale(audio_rotation_3d_x_data,audio_rotation_3d_x_multiplier)
audio_rotation_3d_x=str()
#@markdown ___audio_rotation_3d_y___: Positive rotation_3d_y pans the camera to the right.
audio_rotation_3d_y_preped=audio_rotation_3d_y_data = np.array([], dtype=np.float64)
audio_rotation_3d_y_frequency = "Lower Midrange" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_rotation_3d_y_multiplier = 0.5 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_rotation_3d_y_soften = True #@param {type: "boolean" }
audio_rotation_3d_y_kernel_size=216 #@param{type: 'integer'}
audio_rotation_3d_y_data =  locals()[rangeLookup(audio_rotation_3d_y_frequency)]
if audio_rotation_3d_y_soften == True and audio_rotation_3d_y_kernel_size != '':
  audio_rotation_3d_y_data = smoothing(audio_rotation_3d_y_data,audio_rotation_3d_y_kernel_size)
audio_rotation_3d_y_data = normAndScale(audio_rotation_3d_y_data,audio_rotation_3d_y_multiplier)
audio_rotation_3d_y=str()
#@markdown ___audio_rotation_3d_z___: Positive rotation_3d_z rolls the camera clockwise.
audio_rotation_3d_z_preped=audio_rotation_3d_z_data = np.array([], dtype=np.float64)
audio_rotation_3d_z_frequency = "Upper Midrange" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_rotation_3d_z_multiplier = 0.5 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_rotation_3d_z_soften = True #@param {type: "boolean" }
audio_rotation_3d_z_kernel_size=216 #@param{type: 'integer'}
audio_rotation_3d_z_data =  locals()[rangeLookup(audio_rotation_3d_z_frequency)]
if audio_rotation_3d_z_soften == True and audio_rotation_3d_z_kernel_size != '':
  audio_rotation_3d_z_data = smoothing(audio_rotation_3d_z_data,audio_rotation_3d_z_kernel_size)
audio_rotation_3d_z_data = normAndScale(audio_rotation_3d_z_data,audio_rotation_3d_z_multiplier)
audio_rotation_3d_z=str()

###
audio_zoom=MakeDataStr(audio_zoom_data,"audio_zoom")
audio_angle=MakeDataStr(audio_angle_data,"audio_angle")

audio_rotation_3d_z=MakeDataStr(audio_rotation_3d_z_data,"audio_rotation_3d_z")
audio_rotation_3d_y=MakeDataStr(audio_rotation_3d_y_data,"audio_rotation_3d_y")
audio_rotation_3d_x=MakeDataStr(audio_rotation_3d_x_data,"audio_rotation_3d_x")

audio_translation_z=MakeDataStr(audio_translation_z_data,"audio_translation_z")
audio_translation_y=MakeDataStr(audio_translation_y_data,"audio_translation_y")
audio_translation_x=MakeDataStr(audio_translation_x_data,"audio_translation_x")
#@markdown ---
#@markdown ___audio_cut_overview && audio_cut_innercut___: experimental cut scheduling. multiplier is cuts total.
audio_cut_innercut_preped=audio_cut_innercut_data=audio_cut_overview_preped=audio_cut_overview_data = np.array([], dtype=np.float64)
audio_cut_overview_frequency = "Full" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_cut_overview_multiplier = 24 #@param [12,16,24,32] {type:"raw"}
audio_cut_overview_soften = True #@param {type: "boolean" }
audio_cut_overview_kernel_size=20 #@param{type: 'integer'}
audio_cut_overview_buckets=216 #@param{type: 'integer'}
audio_cut_overview_data =  locals()[rangeLookup(audio_cut_overview_frequency)]
if audio_cut_overview_soften == True and audio_cut_overview_kernel_size != '':
  audio_cut_overview_data = smoothing(audio_cut_overview_data,audio_cut_overview_kernel_size)
audio_cut_overview_data = normAndScalePositive(audio_cut_overview_data,1)
audio_cut_innercut=audio_cut_overview=str() 
audio_cut_overview = "[12]*400+[4]*600"
audio_cut_innercut = "[4]*400+[12]*600"
audio_cut_overview,audio_cut_innercut = MakeCutDataStr(audio_cut_overview_data,audio_cut_overview_multiplier,audio_cut_overview_buckets)
###



###
#@markdown ---
#@markdown ___Blooming___: Blur + Over Exposure
audio_fx_blooming_enabled = True #@param {type: "boolean" }
audio_fx_blooming_preped=audio_fx_blooming_data = np.array([], dtype=np.float64)
audio_fx_blooming_frequency = "Full" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_fx_blooming_multiplier = 0.05 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_fx_blooming_soften = False #@param {type: "boolean" }
audio_fx_blooming_kernel_size=216 #@param{type: 'integer'}

if audio_fx_blooming_enabled == True:
  audio_fx_blooming_data =  locals()[rangeLookup(audio_fx_blooming_frequency)]
  if audio_fx_blooming_soften == True and audio_fx_blooming_kernel_size != '':
    audio_fx_blooming_data = smoothing(audio_fx_blooming_data,audio_fx_blooming_kernel_size)
  audio_fx_blooming_data = normAndScale(audio_fx_blooming_data,audio_fx_blooming_multiplier)
  audio_fx_blooming=str()
  plotGraph(audio_fx_blooming_data,'audio_fx_blooming_data')
#@markdown ___Polar Waveform___: Overlay Audio Waveform in a polar pattern
audio_fx_polarwaveform_enabled = False #@param {type: "boolean" }
audio_fx_polarwaveform_preped=audio_fx_polarwaveform_data = np.array([], dtype=np.float64)
audio_fx_polarwaveform_frequency = "Full" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_fx_polarwaveform_multiplier = 1 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_fx_polarwaveform_soften = False #@param {type: "boolean" }
audio_fx_polarwaveform_kernel_size=216 #@param{type: 'integer'}
#audio_fx_polarwaveform_kernel_size=int(np.floor(max_iterations*(2/3))) #@param{type: 'integer'}
audio_fx_polarwaveform_zooming = True #@param {type: "boolean" }
if audio_fx_polarwaveform_enabled == True:
  audio_fx_polarwaveform_data =  locals()[rangeLookup(audio_fx_polarwaveform_frequency)]
  if audio_fx_polarwaveform_soften == True and audio_fx_polarwaveform_kernel_size != '':
    audio_fx_polarwaveform_data = smoothing(audio_fx_polarwaveform_data,audio_fx_polarwaveform_kernel_size)
  audio_fx_polarwaveform_data = normAndScale(audio_fx_polarwaveform_data,audio_fx_polarwaveform_multiplier)
  audio_fx_polarwaveform=str()
  plotGraph(audio_fx_polarwaveform_data,'audio_fx_polarwaveform_data')
#@markdown ___Frequency Splat___: Overlay semirandom circles based on audio data
audio_fx_splat_enabled = True #@param {type: "boolean" }
audio_fx_splat_preped=audio_fx_splat_data = np.array([], dtype=np.float64)
audio_fx_splat_frequency = "Full" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_fx_splat_multiplier = 1 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_fx_splat_soften = False #@param {type: "boolean" }
audio_fx_splat_kernel_size=216 #@param{type: 'integer'}
if audio_fx_splat_enabled == True:
  audio_fx_splat_data =  locals()[rangeLookup(audio_fx_splat_frequency)]
  if audio_fx_splat_soften == True and audio_fx_splat_kernel_size != '':
    audio_fx_splat_data = smoothing(audio_fx_splat_data,audio_fx_splat_kernel_size)
  audio_fx_splat_data = normAndScale(audio_fx_splat_data,audio_fx_splat_multiplier)
  audio_fx_splat=str()
  plotGraph(audio_fx_splat_data,'audio_fx_splat_data')
#@markdown ___3dVoxel Torus___: Overlay Audio Waveform in a 3dVoxel Torus
audio_fx_3dtorus_enabled = False #@param {type: "boolean" }
audio_fx_3dtorus_preped=audio_fx_3dtorus_data = np.array([], dtype=np.float64)
audio_fx_3dtorus_frequency = "Full" #@param ["Full", "Sub Bass","Bass", "Lower Midrange", "Midrange","Upper Midrange", "Presence", "Brilliance","3StepLFO","Sine","Saw","Square"]
audio_fx_3dtorus_multiplier = 1 #@param [0.05, 0.25 ,0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {type:"raw"}
audio_fx_3dtorus_soften = False #@param {type: "boolean" }
audio_fx_3dtorus_kernel_size=216 #@param{type: 'integer'}
audio_fx_3dtorus_zooming = True #@param {type: "boolean" }
if audio_fx_3dtorus_enabled == True:
  audio_fx_3dtorus_data =  locals()[rangeLookup(audio_fx_3dtorus_frequency)]
  if audio_fx_3dtorus_soften == True and audio_fx_3dtorus_kernel_size != '':
    audio_fx_3dtorus_data = smoothing(audio_fx_3dtorus_data,audio_fx_3dtorus_kernel_size)
  audio_fx_3dtorus_data = normAndScale(audio_fx_3dtorus_data,audio_fx_3dtorus_multiplier)
  audio_fx_3dtorus=str()
  plotGraph(audio_fx_3dtorus_data,'audio_fx_3dtorus_data')

#@markdown # ___Text Prompt KeyFrame Recommender___ #
audio_fx_TPKR_enabled = True #@param {type: "boolean" }
#audio_fx_TPKR_data = np.array([], dtype=np.float64)
audio_fx_TPKR_promptToBeatRatio_beats = 1 #@param {type: "number" }
audio_fx_TPKR_promptToBeatRatio_measure = 32 #@param {type: "number" }
audio_fx_TPKR_type = "PredominantLocalPulse" #@param ["BeatDetection", "Superflux","PredominantLocalPulse","Backtracked"]
#@markdown  ___@NeuralismAI Generative Art prompt generator___ 
audio_fx_TPKR_generator_enabled = True #@param {type: "boolean" }
audio_fx_TPKR_generator_theme_enabled = True #@param {type: "boolean" }
audio_fx_TPKR_generator_theme_string = '"galaxies, stars, nebulae:6", "bright, alien, futuristic, spaceship and astro theme:2", "neon acrylic paint on black matte, by Ilya Kuvshinov:1", "Cel shading, by Seb McKinnon and Matt Hughes and Fabio Listrani and Alfons Maria Mucha and Ellen Jewett and Nicola Samori:3", "Bokeh:1", "Ultrawide Lens:2","Ray tracing, by Hanna-Barbera:-2", "car:-1", "dof:-1", "blur:-1", "lines:-1", "streaks:-1", "globe:-1", "painting:-1", "photorealism:-1"' #@param {type:"raw"}
promptList = []
VQGAN_prompt = False # param {type:"boolean"}
n_adjectives = 2 #@param {type:"integer"}
n_artists =   1#@param {type:"integer"}
n_colors = 1 #@param {type:"integer"}
n_art_styles =  1#@param {type:"integer"}
n_things =  1#@param {type:"integer"}
Animals = True #@param {type:"boolean"}
Locations = True #@param {type:"boolean"}
Shapes = True #@param {type:"boolean"}

if audio_fx_TPKR_enabled == True:
  audio_fx_TPKR_data= {}
  audio_y, audio_sr = librosa.load(init_audio)
  audio_fx_TPKR_data  = {"y":audio_y, "sr":audio_sr}
  audio_fx_TPKR=str()
  beatTime=plotBeatGraph(audio_fx_TPKR_type,audio_fx_TPKR_data,'audio_fx_TPKR_data',fps)
  audio_fx_TPKR_counter=audio_fx_TPKR_promptToBeatRatio_beats
  beatTimeIter={0:''}
  beatTimeKeys=list(beatTime.keys())
  #for btime_idx,btime in beatTime.items():
  bt_cnr=0
  for i in range(len(beatTimeKeys)):
    if audio_fx_TPKR_counter == audio_fx_TPKR_promptToBeatRatio_beats:
      #beatTimeIter[btime_idx].append(btime)
      #print(beatTimeKeys[i])
      beatTimeIter[beatTimeKeys[i]] = ''
    if audio_fx_TPKR_counter < audio_fx_TPKR_promptToBeatRatio_measure:
      audio_fx_TPKR_counter += audio_fx_TPKR_promptToBeatRatio_beats
    else:
      audio_fx_TPKR_counter = 1
  beatTime=beatTimeIter
  if audio_fx_TPKR_generator_enabled == True:
    %shell git clone https://github.com/sanzelda/prompt_gen.git /content/prompt_gen || echo "clone failed"
    import random
    amount_of_prompts = len(beatTime)
    make_random_prompt(amount_of_prompts)
    elpromptList=[[el] for el in promptList]
    TPKR_data = dict(zip(beatTime.keys(), elpromptList))
    if audio_fx_TPKR_generator_theme_enabled == True:
      tpkr_string_list=[]
      for tpkr_string in audio_fx_TPKR_generator_theme_string.split(','):
        tpkr_string_list.append(tpkr_string.strip('"').strip(','))
      for akey in TPKR_data.keys():
        TPKR_data[akey] = TPKR_data[akey] + tpkr_string_list
  else:
    TPKR_data = beatTime
  #plotGraph(TPKR_data,'TPKR_data')
  print("text_prompts = ",TPKR_data)
  text_prompts = TPKR_data

# %%
# !! {"metadata":{
# !!   "id": "AnimSetTop"
# !! }}
"""
### Animation Settings
"""

# %%
# !! {"metadata":{
# !!   "id": "AnimSettings"
# !! }}
#@markdown ####**Animation Mode:**
animation_mode = '3D' #@param ['None', '2D', '3D', 'Video Input'] {type:'string'}
#@markdown *For animation, you probably want to turn `cutn_batches` to 1 to make it quicker.*


#@markdown ---

#@markdown ####**Video Input Settings:**
if is_colab:
    video_init_path = "/content/drive/MyDrive/init.mp4" #@param {type: 'string'}
else:
    video_init_path = "init.mp4" #@param {type: 'string'}
extract_nth_frame = 2 #@param {type: 'number'}
persistent_frame_output_in_batch_folder = True #@param {type: 'boolean'}
video_init_seed_continuity = False #@param {type: 'boolean'}
#@markdown #####**Video Optical Flow Settings:**
video_init_flow_warp = True #@param {type: 'boolean'}
# Call optical flow from video frames and warp prev frame with flow
video_init_flow_blend =  0.999#@param {type: 'number'} #0 - take next frame, 1 - take prev warped frame
video_init_check_consistency = False #Insert param here when ready
video_init_blend_mode = "optical flow" #@param ['None', 'linear', 'optical flow']
# Call optical flow from video frames and warp prev frame with flow
if animation_mode == "Video Input":
    if persistent_frame_output_in_batch_folder or (not is_colab): #suggested by Chris the Wizard#8082 at discord
        videoFramesFolder = f'{batchFolder}/videoFrames'
    else:
        videoFramesFolder = f'/content/videoFrames'
    createPath(videoFramesFolder)
    print(f"Exporting Video Frames (1 every {extract_nth_frame})...")
    try:
        for f in pathlib.Path(f'{videoFramesFolder}').glob('*.jpg'):
            f.unlink()
    except:
        print('')
    vf = f'select=not(mod(n\,{extract_nth_frame}))'
    if os.path.exists(video_init_path):
        subprocess.run(['ffmpeg', '-i', f'{video_init_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{videoFramesFolder}/%04d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    else: 
        print(f'\nWARNING!\n\nVideo not found: {video_init_path}.\nPlease check your video path.\n')
    #!ffmpeg -i {video_init_path} -vf {vf} -vsync vfr -q:v 2 -loglevel error -stats {videoFramesFolder}/%04d.jpg


#@markdown ---

#@markdown ####**2D Animation Settings:**
#@markdown `zoom` is a multiplier of dimensions, 1 is no zoom.
#@markdown All rotations are provided in degrees.

key_frames = True #@param {type:"boolean"}
max_frames = max_iterations #@ param {type:"number"}

if animation_mode == "Video Input":
    max_frames = len(glob(f'{videoFramesFolder}/*.jpg'))

interp_spline = 'Linear' #Do not change, currently will not look good. param ['Linear','Quadratic','Cubic']{type:"string"}
angle = audio_angle #@param {type:"string"}
zoom = audio_zoom #@param {type:"string"}
translation_x = audio_translation_x #@param {type:"string"}
translation_y = audio_translation_y #@param {type:"string"}
translation_z = audio_translation_z #@param {type:"string"}
rotation_3d_x = audio_rotation_3d_x #@param {type:"string"}
rotation_3d_y = audio_rotation_3d_y #@param {type:"string"}
rotation_3d_z = audio_rotation_3d_z #@param {type:"string"}
midas_depth_model = "dpt_large"#@param {type:"string"}
midas_weight = 0.3#@param {type:"number"}
near_plane = 200#@param {type:"number"}
far_plane = 10000#@param {type:"number"}
fov = 40#@param {type:"number"}
padding_mode = 'border'#@param {type:"string"}
sampling_mode = 'bicubic'#@param {type:"string"}

#======= TURBO MODE
#@markdown ---
#@markdown ####**Turbo Mode (3D anim only):**
#@markdown (Starts after frame 10,) skips diffusion steps and just uses depth map to warp images for skipped frames.
#@markdown Speeds up rendering by 2x-4x, and may improve image coherence between frames.
#@markdown For different settings tuned for Turbo Mode, refer to the original Disco-Turbo Github: https://github.com/zippy731/disco-diffusion-turbo

turbo_mode = True #@param {type:"boolean"}
turbo_steps = "3" #@param ["2","3","4","5","6"] {type:"string"}
turbo_preroll = 10 # frames

#insist turbo be used only w 3d anim.
if turbo_mode and animation_mode != '3D':
    print('=====')
    print('Turbo mode only available with 3D animations. Disabling Turbo.')
    print('=====')
    turbo_mode = False

#@markdown ---

#@markdown ####**Coherency Settings:**
#@markdown `frame_scale` tries to guide the new frame to looking like the old one. A good default is 1500.
frames_scale = 2000 #@param{type: 'integer'}
#@markdown `frame_skip_steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
frames_skip_steps = '60%' #@param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}

#@markdown ####**Video Init Coherency Settings:**
#@markdown `frame_scale` tries to guide the new frame to looking like the old one. A good default is 1500.
video_init_frames_scale = 15000 #@param{type: 'integer'}
#@markdown `frame_skip_steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
video_init_frames_skip_steps = '70%' #@param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}

#======= VR MODE
#@markdown ---
#@markdown ####**VR Mode (3D anim only):**
#@markdown Enables stereo rendering of left/right eye views (supporting Turbo) which use a different (fish-eye) camera projection matrix.   
#@markdown Note the images you're prompting will work better if they have some inherent wide-angle aspect
#@markdown The generated images will need to be combined into left/right videos. These can then be stitched into the VR180 format.
#@markdown Google made the VR180 Creator tool but subsequently stopped supporting it. It's available for download in a few places including https://www.patrickgrunwald.de/vr180-creator-download
#@markdown The tool is not only good for stitching (videos and photos) but also for adding the correct metadata into existing videos, which is needed for services like YouTube to identify the format correctly.
#@markdown Watching YouTube VR videos isn't necessarily the easiest depending on your headset. For instance Oculus have a dedicated media studio and store which makes the files easier to access on a Quest https://creator.oculus.com/manage/mediastudio/
#@markdown 
#@markdown The command to get ffmpeg to concat your frames for each eye is in the form: `ffmpeg -framerate 15 -i frame_%4d_l.png l.mp4` (repeat for r)

vr_mode = False #@param {type:"boolean"}
#@markdown `vr_eye_angle` is the y-axis rotation of the eyes towards the center
vr_eye_angle = 0.5 #@param{type:"number"}
#@markdown interpupillary distance (between the eyes)
vr_ipd = 5.0 #@param{type:"number"}

#insist VR be used only w 3d anim.
if vr_mode and animation_mode != '3D':
    print('=====')
    print('VR mode only available with 3D animations. Disabling VR.')
    print('=====')
    vr_mode = False


def parse_key_frames(string, prompt_parser=None):
    """Given a string representing frame numbers paired with parameter values at that frame,
    return a dictionary with the frame numbers as keys and the parameter values as the values.

    Parameters
    ----------
    string: string
        Frame numbers paired with parameter values at that frame number, in the format
        'framenumber1: (parametervalues1), framenumber2: (parametervalues2), ...'
    prompt_parser: function or None, optional
        If provided, prompt_parser will be applied to each string of parameter values.
    
    Returns
    -------
    dict
        Frame numbers as keys, parameter values at that frame number as values

    Raises
    ------
    RuntimeError
        If the input string does not match the expected format.
    
    Examples
    --------
    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)")
    {10: 'Apple: 1| Orange: 0', 20: 'Apple: 0| Orange: 1| Peach: 1'}

    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)", prompt_parser=lambda x: x.lower()))
    {10: 'apple: 1| orange: 0', 20: 'apple: 0| orange: 1| peach: 1'}
    """
    import re
    pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param

    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames

def get_inbetweens(key_frames, integer=False):
    """Given a dict with frame numbers as keys and a parameter value as values,
    return a pandas Series containing the value of the parameter at every frame from 0 to max_frames.
    Any values not provided in the input dict are calculated by linear interpolation between
    the values of the previous and next provided frames. If there is no previous provided frame, then
    the value is equal to the value of the next provided frame, or if there is no next provided frame,
    then the value is equal to the value of the previous provided frame. If no frames are provided,
    all frame values are NaN.

    Parameters
    ----------
    key_frames: dict
        A dict with integer frame numbers as keys and numerical values of a particular parameter as values.
    integer: Bool, optional
        If True, the values of the output series are converted to integers.
        Otherwise, the values are floats.
    
    Returns
    -------
    pd.Series
        A Series with length max_frames representing the parameter values for each frame.
    
    Examples
    --------
    >>> max_frames = 5
    >>> get_inbetweens({1: 5, 3: 6})
    0    5.0
    1    5.0
    2    5.5
    3    6.0
    4    6.0
    dtype: float64

    >>> get_inbetweens({1: 5, 3: 6}, integer=True)
    0    5
    1    5
    2    5
    3    6
    4    6
    dtype: int64
    """
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)
    
    interp_method = interp_spline

    if interp_method == 'Cubic' and len(key_frames.items()) <=3:
      interp_method = 'Quadratic'
    
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
      interp_method = 'Linear'
      
    
    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
    # key_frame_series = key_frame_series.interpolate(method=intrp_method,order=1, limit_direction='both')
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(),limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

def split_prompts(prompts):
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series

if key_frames:
    try:
        angle_series = get_inbetweens(parse_key_frames(angle))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `angle` correctly for key frames.\n"
            "Attempting to interpret `angle` as "
            f'"0: ({angle})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        angle = f"0: ({angle})"
        angle_series = get_inbetweens(parse_key_frames(angle))

    try:
        zoom_series = get_inbetweens(parse_key_frames(zoom))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `zoom` correctly for key frames.\n"
            "Attempting to interpret `zoom` as "
            f'"0: ({zoom})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        zoom = f"0: ({zoom})"
        zoom_series = get_inbetweens(parse_key_frames(zoom))

    try:
        translation_x_series = get_inbetweens(parse_key_frames(translation_x))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_x` correctly for key frames.\n"
            "Attempting to interpret `translation_x` as "
            f'"0: ({translation_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_x = f"0: ({translation_x})"
        translation_x_series = get_inbetweens(parse_key_frames(translation_x))

    try:
        translation_y_series = get_inbetweens(parse_key_frames(translation_y))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_y` correctly for key frames.\n"
            "Attempting to interpret `translation_y` as "
            f'"0: ({translation_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_y = f"0: ({translation_y})"
        translation_y_series = get_inbetweens(parse_key_frames(translation_y))

    try:
        translation_z_series = get_inbetweens(parse_key_frames(translation_z))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_z` correctly for key frames.\n"
            "Attempting to interpret `translation_z` as "
            f'"0: ({translation_z})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_z = f"0: ({translation_z})"
        translation_z_series = get_inbetweens(parse_key_frames(translation_z))

    try:
        rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_x` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_x` as "
            f'"0: ({rotation_3d_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_x = f"0: ({rotation_3d_x})"
        rotation_3d_x_series = get_inbetweens(parse_key_frames(rotation_3d_x))

    try:
        rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_y` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_y` as "
            f'"0: ({rotation_3d_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_y = f"0: ({rotation_3d_y})"
        rotation_3d_y_series = get_inbetweens(parse_key_frames(rotation_3d_y))

    try:
        rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z))
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_z` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_z` as "
            f'"0: ({rotation_3d_z})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_z = f"0: ({rotation_3d_z})"
        rotation_3d_z_series = get_inbetweens(parse_key_frames(rotation_3d_z))

else:
    angle = float(angle)
    zoom = float(zoom)
    translation_x = float(translation_x)
    translation_y = float(translation_y)
    translation_z = float(translation_z)
    rotation_3d_x = float(rotation_3d_x)
    rotation_3d_y = float(rotation_3d_y)
    rotation_3d_z = float(rotation_3d_z)

# %%
# !! {"metadata":{
# !!   "id": "InstallRAFT"
# !! }}
#@title Install RAFT for Video input animation mode only
#@markdown Run once per session. Doesn't download again if model path exists.
#@markdown Use force download to reload raft models if needed
force_download = False #@param {type:'boolean'}
if animation_mode == 'Video Input':
    try:
        from raft import RAFT
    except:
        if not os.path.exists(os.path.join(PROJECT_DIR, 'RAFT')):
            gitclone('https://github.com/princeton-vl/RAFT', os.path.join(PROJECT_DIR, 'RAFT'))
        sys.path.append(f'{PROJECT_DIR}/RAFT')

    if (not (os.path.exists(f'{root_path}/RAFT/models'))) or force_download:
        createPath(f'{root_path}/RAFT')
        os.chdir(f'{root_path}/RAFT')
        sub_p_res = subprocess.run(['bash', f'{PROJECT_DIR}/RAFT/download_models.sh'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        print(sub_p_res)
        os.chdir(PROJECT_DIR)

# %%
# !! {"metadata":{
# !!   "id": "FlowFns1"
# !! }}
#@title Define optical flow functions for Video input animation mode only
if animation_mode == 'Video Input':
    in_path = videoFramesFolder
    flo_folder = f'{in_path}/out_flo_fwd'
    path = f'{PROJECT_DIR}/RAFT/core'
    import sys
    sys.path.append(f'{PROJECT_DIR}/RAFT/core')
    os.chdir(f'{PROJECT_DIR}/RAFT/core')
    print(os.getcwd())
  
    print("Renaming RAFT core's utils.utils to raftutils.utils (to avoid a naming conflict with AdaBins)")
    if not os.path.exists(f'{PROJECT_DIR}/RAFT/core/raftutils'):
        os.rename(f'{PROJECT_DIR}/RAFT/core/utils', f'{PROJECT_DIR}/RAFT/core/raftutils')
        sub_p_res = subprocess.run(['sed', '-i', 's/from utils.utils/from raftutils.utils/g', f'{PROJECT_DIR}/RAFT/core/corr.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        sub_p_res = subprocess.run(['sed', '-i', 's/from utils.utils/from raftutils.utils/g', f'{PROJECT_DIR}/RAFT/core/raft.py'], stdout=subprocess.PIPE).stdout.decode('utf-8')

    from raftutils.utils import InputPadder
    from raft import RAFT
    from raftutils import flow_viz
    import numpy as np
    import argparse, PIL, cv2
    from PIL import Image
    from tqdm.notebook import tqdm
    from glob import glob
    import torch
  
    args2 = argparse.Namespace()
    args2.small = False
    args2.mixed_precision = True
  
  
    TAG_CHAR = np.array([202021.25], np.float32)
  
    def writeFlow(filename,uv,v=None):
        """ 
        https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/flow_utils.py
        Copyright 2017 NVIDIA CORPORATION
  
        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at
  
            http://www.apache.org/licenses/LICENSE-2.0
  
        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
        
        Write optical flow to file.
        
        If v is None, uv is assumed to contain both u and v channels,
        stacked in depth.
        Original code by Deqing Sun, adapted from Daniel Scharstein.
        """
        nBands = 2
  
        if v is None:
            assert(uv.ndim == 3)
            assert(uv.shape[2] == 2)
            u = uv[:,:,0]
            v = uv[:,:,1]
        else:
            u = uv
  
        assert(u.shape == v.shape)
        height,width = u.shape
        f = open(filename,'wb')
        # write the header
        f.write(TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width*nBands))
        tmp[:,np.arange(width)*2] = u
        tmp[:,np.arange(width)*2 + 1] = v
        tmp.astype(np.float32).tofile(f)
        f.close()
  
    def load_img(img, size):
        img = Image.open(img).convert('RGB').resize(size)
        return torch.from_numpy(np.array(img)).permute(2,0,1).float()[None,...].cuda()
  
    def get_flow(frame1, frame2, model, iters=20):
        padder = InputPadder(frame1.shape)
        frame1, frame2 = padder.pad(frame1, frame2)
        _, flow12 = model(frame1, frame2, iters=iters, test_mode=True)
        flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()
  
        return flow12
  
    def warp_flow(img, flow):
        h, w = flow.shape[:2]
        flow = flow.copy()
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res
  
    def makeEven(_x):
        return _x if (_x % 2 == 0) else _x+1
  
    def fit(img,maxsize=512):
        maxdim = max(*img.size)
        if maxdim>maxsize:
            # if True:
            ratio = maxsize/maxdim
            x,y = img.size
            size = (makeEven(int(x*ratio)),makeEven(int(y*ratio))) 
            img = img.resize(size)
        return img
  
    def warp(frame1, frame2, flo_path, blend=0.5, weights_path=None):
        flow21 = np.load(flo_path)
        frame1pil = np.array(frame1.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
        frame1_warped21 = warp_flow(frame1pil, flow21)
        # frame2pil = frame1pil
        frame2pil = np.array(frame2.convert('RGB').resize((flow21.shape[1],flow21.shape[0])))
    
        if weights_path:
            # TBD
            pass
        else:
            blended_w = frame2pil*(1-blend) + frame1_warped21*(blend)
  
        return  PIL.Image.fromarray(blended_w.astype('uint8'))
  
    in_path = videoFramesFolder
    flo_folder = f'{in_path}/out_flo_fwd'
  
    temp_flo = in_path+'/temp_flo'
    flo_fwd_folder = in_path+'/out_flo_fwd'
    # TBD flow backwards!
  
    os.chdir(PROJECT_DIR)

# %%
# !! {"metadata":{
# !!   "id": "FlowFns2"
# !! }}
#@title Generate optical flow and consistency maps
#@markdown Run once per init video

if animation_mode == "Video Input":
    import gc

    force_flow_generation = False #@param {type:'boolean'}
    in_path = videoFramesFolder
    flo_folder = f'{in_path}/out_flo_fwd'

    if not video_init_flow_warp:
        print('video_init_flow_warp not set, skipping')

    if (animation_mode == 'Video Input') and (video_init_flow_warp):
        flows = glob(flo_folder+'/*.*')
        if (len(flows)>0) and not force_flow_generation:
            print(f'Skipping flow generation:\nFound {len(flows)} existing flow files in current working folder: {flo_folder}.\nIf you wish to generate new flow files, check force_flow_generation and run this cell again.')
    
        if (len(flows)==0) or force_flow_generation:
            frames = sorted(glob(in_path+'/*.*'));
            if len(frames)<2: 
                print(f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.')
            if len(frames)>=2:
        
                raft_model = torch.nn.DataParallel(RAFT(args2))
                print('Loading RAFT checkpoint with weights_only=False -- downloaded via RAFT\'s own '
                      'download_models.sh, not diff_model_map, so there is no pinned hash to verify '
                      'against here.')
                raft_model.load_state_dict(torch.load(f'{root_path}/RAFT/models/raft-things.pth', weights_only=False))
                raft_model = raft_model.module.cuda().eval()
        
                for f in pathlib.Path(f'{flo_fwd_folder}').glob('*.*'):
                    f.unlink()
        
                temp_flo = in_path+'/temp_flo'
                flo_fwd_folder = in_path+'/out_flo_fwd'
        
                createPath(flo_fwd_folder)
                createPath(temp_flo)
        
                # TBD Call out to a consistency checker?
        
                framecount = 0
                for frame1, frame2 in tqdm(zip(frames[:-1], frames[1:]), total=len(frames)-1):
        
                    out_flow21_fn = f"{flo_fwd_folder}/{frame1.split('/')[-1]}"
            
                    frame1 = load_img(frame1, width_height)
                    frame2 = load_img(frame2, width_height)
            
                    flow21 = get_flow(frame2, frame1, raft_model)
                    np.save(out_flow21_fn, flow21)
            
                    if video_init_check_consistency:
                        # TBD
                        pass

                del raft_model 
                gc.collect()

# %%
# !! {"metadata":{
# !!   "id": "ExtraSetTop"
# !! }}
"""
### Extra Settings
 Partial Saves, Advanced Settings, Cutn Scheduling
"""

# %%
# !! {"metadata":{
# !!   "id": "ExtraSettings"
# !! }}
#@markdown ####**Saving:**

intermediate_saves = 0#@param{type: 'raw'}
intermediates_in_subfolder = True #@param{type: 'boolean'}
#@markdown Intermediate steps will save a copy at your specified intervals. You can either format it as a single integer or a list of specific steps 

#@markdown A value of `2` will save a copy at 33% and 66%. 0 will save none.

#@markdown A value of `[5, 9, 34, 45]` will save at steps 5, 9, 34, and 45. (Make sure to include the brackets)


if type(intermediate_saves) is not list:
    if intermediate_saves:
        steps_per_checkpoint = math.floor((steps - skip_steps - 1) // (intermediate_saves+1))
        steps_per_checkpoint = steps_per_checkpoint if steps_per_checkpoint > 0 else 1
        print(f'Will save every {steps_per_checkpoint} steps')
    else:
        steps_per_checkpoint = steps+10
else:
    steps_per_checkpoint = None

if intermediate_saves and intermediates_in_subfolder is True:
    partialFolder = f'{batchFolder}/partials'
    createPath(partialFolder)

#@markdown ---

#@markdown ####**Advanced Settings:**
#@markdown *There are a few extra advanced settings available if you double click this cell.*

#@markdown *Perlin init will replace your init, so uncheck if using one.*

perlin_init = False  #@param{type: 'boolean'}
perlin_mode = 'mixed' #@param ['mixed', 'color', 'gray']
set_seed = 'random_seed' #@param{type: 'string'}
eta = 1.0#@param{type: 'number'}
clamp_grad = True #@param{type: 'boolean'}
clamp_max = 0.05 #@param{type: 'number'}


### EXTRA ADVANCED SETTINGS:
randomize_class = True
clip_denoised = False
fuzzy_prompt = False
rand_mag = 0.05


#@markdown ---

#@markdown ####**Cutn Scheduling:**
#@markdown Format: `[40]*400+[20]*600` = 40 cuts for the first 400 /1000 steps, then 20 for the last 600/1000

#@markdown cut_overview and cut_innercut are cumulative for total cutn on any given step. Overview cuts see the entire image and are good for early structure, innercuts are your standard cutn.

cut_overview = str(audio_cut_overview)    #@param {type: 'string'}  
cut_innercut = str(audio_cut_innercut)  #@param {type: 'string'}  
cut_ic_pow = "[1]*1000" #@param {type: 'string'}  
cut_icgray_p = "[0.2]*400+[0]*600" #@param {type: 'string'}

#@markdown KaliYuga model settings. Refer to [cut_ic_pow](https://ezcharts.miraheze.org/wiki/Category:Cut_ic_pow) as a guide. Values between 1 and 100 all work.
pad_or_pulp_cut_overview = "[15]*100+[15]*100+[12]*100+[12]*100+[6]*100+[4]*100+[2]*200+[0]*200" #@param {type: 'string'}
pad_or_pulp_cut_innercut = "[1]*100+[1]*100+[4]*100+[4]*100+[8]*100+[8]*100+[10]*200+[10]*200" #@param {type: 'string'}
pad_or_pulp_cut_ic_pow = "[12]*300+[12]*100+[12]*50+[12]*50+[10]*100+[10]*100+[10]*300" #@param {type: 'string'}
pad_or_pulp_cut_icgray_p = "[0.87]*100+[0.78]*50+[0.73]*50+[0.64]*60+[0.56]*40+[0.50]*50+[0.33]*100+[0.19]*150+[0]*400" #@param {type: 'string'}

watercolor_cut_overview = "[14]*200+[12]*200+[4]*400+[0]*200" #@param {type: 'string'}
watercolor_cut_innercut = "[2]*200+[4]*200+[12]*400+[12]*200" #@param {type: 'string'}
watercolor_cut_ic_pow = "[12]*300+[12]*100+[12]*50+[12]*50+[10]*100+[10]*100+[10]*300" #@param {type: 'string'}
watercolor_cut_icgray_p = "[0.7]*100+[0.6]*100+[0.45]*100+[0.3]*100+[0]*600" #@param {type: 'string'}

if (diffusion_model in kaliyuga_pixel_art_model_names) or (diffusion_model in kaliyuga_pulpscifi_model_names):
    cut_overview = pad_or_pulp_cut_overview
    cut_innercut = pad_or_pulp_cut_innercut
    cut_ic_pow = pad_or_pulp_cut_ic_pow
    cut_icgray_p = pad_or_pulp_cut_icgray_p
elif diffusion_model in kaliyuga_watercolor_model_names:
    cut_overview = watercolor_cut_overview
    cut_innercut = watercolor_cut_innercut
    cut_ic_pow = watercolor_cut_ic_pow
    cut_icgray_p = watercolor_cut_icgray_p

#@markdown ---

#@markdown ####**Transformation Settings:**
use_vertical_symmetry = False #@param {type:"boolean"}
use_horizontal_symmetry = False #@param {type:"boolean"}
transformation_percent = [0.09] #@param

# %%
# !! {"metadata":{
# !!   "id": "PromptsTop"
# !! }}
"""
### Prompts
`animation_mode: None` will only use the first set. `animation_mode: 2D / Video` will run through them per the set frames and hold on the last one.
"""

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "Prompts"
# !! }}
#@markdown text prompts set by audio tool by default double click to edit code and manually override
#text_prompts =  {
# 0: ["coincidental deprived Great Wall of China pentagon parakeet casket, by Dan Mumford, dutch golden age, anime art wallpaper 4k, trending on artstation:6",
#            "gothich hellscape town in the distance:6",
#            "dark and grimy theme:2",
#            "neon acrylic paint on black matte, by Ilya Kuvshinov:1",
#            "Cel shading, by Ilya Kuvshinov:3",           
#            "Bokeh:1", 
#            "Ultrawide Lens:2","Ray tracing, by Hanna-Barbera:-2",
#            "car:-1", 
#            "dof:-1",
#            "blur:-1", 
#            "lines:-1", "streaks:-1",
#            "globe:-1",
#            "painting:-1", 
#            "photorealism:-1"]
#}
 
image_prompts = {
#    0:[f'{init_image}:2',],
}

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "VlFIa65GttR_"
# !! }}
#@markdown Downloading - https://fonts.google.com/ - 1240 fonts from google. 
!git clone https://github.com/google/fonts.git /content/fonts
#!find /content/fonts -name "*.ttf"


# %%
# !! {"metadata":{
# !!   "id": "T4ipXjASRXGb"
# !! }}
#@title Text Overlay Keyframe Based Scheduling
#@markdown the syntax of the object is:  

#@markdown ```frameNumber : [ "text to print:coordX,coordY:R,G,B,A:Font Size:Font Face" ]```  

#@markdown to find font installed on the system run:  

#@markdown ```%%bash find / -name "*.ttf"```  

#@markdown the functions in this example are purely for testing but aren't needed. Remember that at 12 fps, 1 frame is 1/12 of a second. Before using special characters like emjois make sure your selected font supports them.

## {side_x}x{side_y}
## Width x Height 
##
## frame : [ "text to print:coordX,coordY:R,G,B,A:Font Size:Font Face" ]
## find installed font faces with:
# %%bash find / -name "*.ttf"
def text_size_random():
  return str(np.random.choice(np.arange(32,96,6)))
def text_alpha_random():
  return str(np.random.choice(np.arange(128,255)))
def text_color_random():
  return str(np.random.choice(np.arange(0,255)))
def text_font_random():
  fonts_list=["/content/fonts/apache/satisfy/Satisfy-Regular.ttf",
              "/content/fonts/ofl/rubikmoonrocks/RubikMoonrocks-Regular.ttf",
              "/content/fonts/ofl/greatvibes/GreatVibes-Regular.ttf",
              "/content/fonts/ofl/sacramento/Sacramento-Regular.ttf",
              "/content/fonts/ofl/bangers/Bangers-Regular.ttf",
              "/content/fonts/ofl/pressstart2p/PressStart2P-Regular.ttf",
              "/content/fonts/apache/specialelite/SpecialElite-Regular.ttf",
              "/content/fonts/ofl/gloriahallelujah/GloriaHallelujah.ttf",
              "/content/fonts/ofl/monoton/Monoton-Regular.ttf"]
  return random.choice(fonts_list)
def text_coord_random():
  textX=str( side_x / (np.random.choice( np.arange(2,8,2) ) ) )
  textY=str( side_y / (np.random.choice( np.arange(2,8,2) ) ) )
  return str(textX + "," + textY)

def text_RGBA_random():
  return str(text_color_random() +','+ text_color_random() +','+ text_color_random() +','+ text_alpha_random() )
#https://www.alt-codes.net/music_note_alt_codes.php
wordS = 4
word="♪♫♪♫♫♫♫♪♪♪♪♫♪♫♫♪♪♫♪♪♪"


text1=''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random()
text2=''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random()
text3_pre = ''.join(random.choices(word,k=wordS))+ ":" +  text_coord_random() +":" #+text_RGBA_random()
text3_mid = ":"+ text_size_random() +":" #text_font_random()
text_overlays = {
    0: [  "AUDIORIDER:"+str(side_x/2)+","+str(side_y/2)+":255,25,25,255:46:/content/fonts/apache/specialelite/SpecialElite-Regular.ttf" ],
    1: [  text1 ],
    2: [  text1 ],
    3: [  text1 ],
    4: [  text1 ],
    5: [  text1 ],
    6: [  text1 ],
    7: [  text1 ],
    8: [  text1 ],
    9: [  text1 ],
    10: [  text1 ],
    11: [  text1 ],
    48: [  text2 ],
    49: [  text2 ],
    50: [  text2 ],
    51: [  text2 ],
    52: [  text2 ],
    53: [  text2 ],
    54: [  text2 ],
    55: [  text2 ],
    56: [  text2 ],
    57: [  text2 ],
    58: [  text2 ],
    59: [  text2 ],
    60: [  text2 ],
    96: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],
    97: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],
    98: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],
    99: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],
    100: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],
    101: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],
    102: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],
    103: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],
    104: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],
    105: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],
    106: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],
    107: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],
    108: [  text3_pre + text_RGBA_random() + text3_mid + text_font_random() ],

    128: [  ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    192: [  ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    256: [  ''.join(random.choices(word,k=wordS))+ ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    312: [  ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    350: [ ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    400: [  ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    450: [ ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    500: [  ''.join(random.choices(word,k=wordS))+ ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    550: [ ''.join(random.choices(word,k=wordS))+ ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    600: [  ''.join(random.choices(word,k=wordS))+ ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    650: [ ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    700: [ ''.join(random.choices(word,k=wordS))+ ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    1128: [  ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    1192: [  ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    1256: [  ''.join(random.choices(word,k=wordS))+ ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    1312: [  ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    1350: [ ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    1400: [  ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    1450: [ ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    1500: [  ''.join(random.choices(word,k=wordS))+ ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    1550: [ ''.join(random.choices(word,k=wordS))+ ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    1600: [  ''.join(random.choices(word,k=wordS))+ ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    1650: [ ''.join(random.choices(word,k=wordS)) + ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],
    1700: [ ''.join(random.choices(word,k=wordS))+ ":" +  text_coord_random() +":"+text_RGBA_random() +":"+ text_size_random() +":"+text_font_random() ],

}

# %%
# !! {"metadata":{
# !!   "id": "DiffuseTop"
# !! }}
"""
# 4. Diffuse!
"""

# %%
# !! {"metadata":{
# !!   "id": "DoTheRun"
# !! }}
#@title Do the Run!
#@markdown `n_batches` ignored with animation modes.
display_rate = 20 #@param{type: 'number'}
n_batches = 1 #@param{type: 'number'}

if animation_mode == 'Video Input':
    steps = video_init_steps

#Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000//steps)*steps if steps < 1000 else steps
model_config.update({
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
})

batch_size = 1 

def move_files(start_num, end_num, old_folder, new_folder):
    for i in range(start_num, end_num):
        old_file = old_folder + f'/{batch_name}({batchNum})_{i:04}.png'
        new_file = new_folder + f'/{batch_name}({batchNum})_{i:04}.png'
        os.rename(old_file, new_file)

#@markdown ---


resume_run = False #@param{type: 'boolean'}
run_to_resume = 'latest' #@param{type: 'string'}
resume_from_frame = 'latest' #@param{type: 'string'}
retain_overwritten_frames = False #@param{type: 'boolean'}
if retain_overwritten_frames:
    retainFolder = f'{batchFolder}/retained'
    createPath(retainFolder)


skip_step_ratio = int(frames_skip_steps.rstrip("%")) / 100
calc_frames_skip_steps = math.floor(steps * skip_step_ratio)

if animation_mode == 'Video Input':
    frames = sorted(glob(in_path+'/*.*'));
    if len(frames)==0: 
        sys.exit("ERROR: 0 frames found.\nPlease check your video input path and rerun the video settings cell.")
    flows = glob(flo_folder+'/*.*')
    if (len(flows)==0) and video_init_flow_warp:
        sys.exit("ERROR: 0 flow files found.\nPlease rerun the flow generation cell.")

if steps <= calc_frames_skip_steps:
    sys.exit("ERROR: You can't skip more steps than your total steps")

if resume_run:
    if run_to_resume == 'latest':
        try:
            batchNum
        except:
            batchNum = len(glob(f"{batchFolder}/{batch_name}(*)_settings.txt"))-1
    else:
        batchNum = int(run_to_resume)
    if resume_from_frame == 'latest':
        start_frame = len(glob(batchFolder+f"/{batch_name}({batchNum})_*.png"))
        if animation_mode != '3D' and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
            start_frame = start_frame - (start_frame % int(turbo_steps))
    else:
        start_frame = int(resume_from_frame)+1
        if animation_mode != '3D' and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
            start_frame = start_frame - (start_frame % int(turbo_steps))
        if retain_overwritten_frames is True:
            existing_frames = len(glob(batchFolder+f"/{batch_name}({batchNum})_*.png"))
            frames_to_save = existing_frames - start_frame
            print(f'Moving {frames_to_save} frames to the Retained folder')
            move_files(start_frame, existing_frames, batchFolder, retainFolder)
else:
    start_frame = 0
    batchNum = len(glob(batchFolder+"/*.txt"))
    while os.path.isfile(f"{batchFolder}/{batch_name}({batchNum})_settings.txt") or os.path.isfile(f"{batchFolder}/{batch_name}-{batchNum}_settings.txt"):
        batchNum += 1

print(f'Starting Run: {batch_name}({batchNum}) at frame {start_frame}')

if set_seed == 'random_seed':
    random.seed()
    seed = random.randint(0, 2**32)
    # print(f'Using seed: {seed}')
else:
    seed = int(set_seed)

args = {
    'batchNum': batchNum,
    'prompts_series':split_prompts(text_prompts) if text_prompts else None,
    'image_prompts_series':split_prompts(image_prompts) if image_prompts else None,
    'seed': seed,
    'display_rate':display_rate,
    'n_batches':n_batches if animation_mode == 'None' else 1,
    'batch_size':batch_size,
    'batch_name': batch_name,
    'steps': steps,
    'diffusion_sampling_mode': diffusion_sampling_mode,
    'width_height': width_height,
    'clip_guidance_scale': clip_guidance_scale,
    'tv_scale': tv_scale,
    'range_scale': range_scale,
    'sat_scale': sat_scale,
    'cutn_batches': cutn_batches,
    'init_image': init_image,
    'init_scale': init_scale,
    'skip_steps': skip_steps,
    'side_x': side_x,
    'side_y': side_y,
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
    'animation_mode': animation_mode,
    'video_init_path': video_init_path,
    'extract_nth_frame': extract_nth_frame,
    'video_init_seed_continuity': video_init_seed_continuity,
    'key_frames': key_frames,
    'max_frames': max_frames if animation_mode != "None" else 1,
    'interp_spline': interp_spline,
    'start_frame': start_frame,
    'angle': angle,
    'zoom': zoom,
    'translation_x': translation_x,
    'translation_y': translation_y,
    'translation_z': translation_z,
    'rotation_3d_x': rotation_3d_x,
    'rotation_3d_y': rotation_3d_y,
    'rotation_3d_z': rotation_3d_z,
    'midas_depth_model': midas_depth_model,
    'midas_weight': midas_weight,
    'near_plane': near_plane,
    'far_plane': far_plane,
    'fov': fov,
    'padding_mode': padding_mode,
    'sampling_mode': sampling_mode,
    'angle_series':angle_series,
    'zoom_series':zoom_series,
    'translation_x_series':translation_x_series,
    'translation_y_series':translation_y_series,
    'translation_z_series':translation_z_series,
    'rotation_3d_x_series':rotation_3d_x_series,
    'rotation_3d_y_series':rotation_3d_y_series,
    'rotation_3d_z_series':rotation_3d_z_series,
    'frames_scale': frames_scale,
    'skip_step_ratio': skip_step_ratio,
    'calc_frames_skip_steps': calc_frames_skip_steps,
    'text_prompts': text_prompts,
    'image_prompts': image_prompts,
    'cut_overview': eval(cut_overview),
    'cut_innercut': eval(cut_innercut),
    'cut_ic_pow': eval(cut_ic_pow),
    'cut_icgray_p': eval(cut_icgray_p),
    'intermediate_saves': intermediate_saves,
    'intermediates_in_subfolder': intermediates_in_subfolder,
    'steps_per_checkpoint': steps_per_checkpoint,
    'perlin_init': perlin_init,
    'perlin_mode': perlin_mode,
    'set_seed': set_seed,
    'eta': eta,
    'clamp_grad': clamp_grad,
    'clamp_max': clamp_max,
    'skip_augs': skip_augs,
    'randomize_class': randomize_class,
    'clip_denoised': clip_denoised,
    'fuzzy_prompt': fuzzy_prompt,
    'rand_mag': rand_mag,
    'turbo_mode':turbo_mode,
    'turbo_steps':turbo_steps,
    'turbo_preroll':turbo_preroll,
    'use_vertical_symmetry': use_vertical_symmetry,
    'use_horizontal_symmetry': use_horizontal_symmetry,
    'transformation_percent': transformation_percent,
    #video init settings
    'video_init_steps': video_init_steps,
    'video_init_clip_guidance_scale': video_init_clip_guidance_scale,
    'video_init_tv_scale': video_init_tv_scale,
    'video_init_range_scale': video_init_range_scale,
    'video_init_sat_scale': video_init_sat_scale,
    'video_init_cutn_batches': video_init_cutn_batches,
    'video_init_skip_steps': video_init_skip_steps,
    'video_init_frames_scale': video_init_frames_scale,
    'video_init_frames_skip_steps': video_init_frames_skip_steps,
    #warp settings
    'video_init_flow_warp':video_init_flow_warp,
    'video_init_flow_blend':video_init_flow_blend,
    'video_init_check_consistency':video_init_check_consistency,
    'video_init_blend_mode':video_init_blend_mode
}

if animation_mode == 'Video Input':
    # This isn't great in terms of what will get saved to the settings.. but it should work.
    args['steps'] = args['video_init_steps']
    args['clip_guidance_scale'] = args['video_init_clip_guidance_scale']
    args['tv_scale'] = args['video_init_tv_scale']
    args['range_scale'] = args['video_init_range_scale']
    args['sat_scale'] = args['video_init_sat_scale']
    args['cutn_batches'] = args['video_init_cutn_batches']
    args['skip_steps'] = args['video_init_skip_steps']
    args['frames_scale'] = args['video_init_frames_scale']
    args['frames_skip_steps'] = args['video_init_frames_skip_steps']

args = SimpleNamespace(**args)

print('Prepping model...')
model, diffusion = create_model_and_diffusion(**model_config)
if diffusion_model == 'custom':
    print(f'Loading custom checkpoint from {custom_path} with weights_only=False and no pinned '
          f'hash to verify against -- only point this at a checkpoint you already trust.')
    model.load_state_dict(torch.load(custom_path, map_location='cpu', weights_only=False))
else:
    model.load_state_dict(torch.load(f'{model_path}/{get_model_filename(diffusion_model)}', map_location='cpu', weights_only=False))
model.requires_grad_(False).eval().to(device)
for name, param in model.named_parameters():
    if 'qkv' in name or 'norm' in name or 'proj' in name:
        param.requires_grad_()
if model_config['use_fp16']:
    model.convert_to_fp16()

gc.collect()
torch.cuda.empty_cache()
try:
    do_run()
except KeyboardInterrupt:
    pass
finally:
    print('Seed used:', seed)
    gc.collect()
    torch.cuda.empty_cache()

# %%
# !! {"metadata":{
# !!   "id": "CreateVidTop"
# !! }}
"""
# 5. Create the video
"""

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "CreateVid"
# !! }}
import PIL
# @title ### **Create video**
#@markdown Video file will save in the same folder as your images.
from tqdm.notebook import trange
skip_video_for_run_all = False #@param {type: 'boolean'}

if animation_mode == 'Video Input':
    frames = sorted(glob(in_path+'/*.*'));
    if len(frames)==0: 
        sys.exit("ERROR: 0 frames found.\nPlease check your video input path and rerun the video settings cell.")
    flows = glob(flo_folder+'/*.*')
    if (len(flows)==0) and video_init_flow_warp:
        sys.exit("ERROR: 0 flow files found.\nPlease rerun the flow generation cell.")

blend =  0.5#@param {type: 'number'}
video_init_check_consistency = False #@param {type: 'boolean'}
if skip_video_for_run_all == True:
    print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')

else:
    # import subprocess in case this cell is run without the above cells
    import subprocess
    from base64 import b64encode

    latest_run = batchNum

    folder = batch_name #@param
    run = latest_run #@param
    final_frame = 'final_frame'


    init_frame = 1#@param {type:"number"} This is the frame where the video will start
    last_frame = final_frame#@param {type:"number"} You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.
    fps = 12#@param {type:"number"}
    # view_video_in_cell = True #@param {type: 'boolean'}

    frames = []
    # tqdm.write('Generating video...')

    if last_frame == 'final_frame':
        last_frame = len(glob(batchFolder+f"/{folder}({run})_*.png"))
        print(f'Total frames: {last_frame}')

    image_path = f"{outDirPath}/{folder}/{folder}({run})_%04d.png"
    filepath = f"{outDirPath}/{folder}/{folder}({run}).mp4"

    if (video_init_blend_mode == 'optical flow') and (animation_mode == 'Video Input'):
        image_path = f"{outDirPath}/{folder}/flow/{folder}({run})_%04d.png"
        filepath = f"{outDirPath}/{folder}/{folder}({run})_flow.mp4"
        if last_frame == 'final_frame':
            last_frame = len(glob(batchFolder+f"/flow/{folder}({run})_*.png"))
        flo_out = batchFolder+f"/flow"
        createPath(flo_out)
        frames_in = sorted(glob(batchFolder+f"/{folder}({run})_*.png"))
        shutil.copy(frames_in[0], flo_out)
        for i in trange(init_frame, min(len(frames_in), last_frame)):
            frame1_path = frames_in[i-1]
            frame2_path = frames_in[i]
  
            frame1 = PIL.Image.open(frame1_path)
            frame2 = PIL.Image.open(frame2_path)
            frame1_stem = f"{(int(frame1_path.split('/')[-1].split('_')[-1][:-4])+1):04}.jpg"
            flo_path = f"/{flo_folder}/{frame1_stem}.npy"
            weights_path = None
            if video_init_check_consistency:
                # TBD
                pass
            warp(frame1, frame2, flo_path, blend=blend, weights_path=weights_path).save(batchFolder+f"/flow/{folder}({run})_{i:04}.png")
    if video_init_blend_mode == 'linear':
        image_path = f"{outDirPath}/{folder}/blend/{folder}({run})_%04d.png"
        filepath = f"{outDirPath}/{folder}/{folder}({run})_blend.mp4"
        if last_frame == 'final_frame':
            last_frame = len(glob(batchFolder+f"/blend/{folder}({run})_*.png"))
        blend_out = batchFolder+f"/blend"
        createPath(blend_out)
        frames_in = glob(batchFolder+f"/{folder}({run})_*.png")
        shutil.copy(frames_in[0], blend_out)
        for i in trange(1, len(frames_in)):
            frame1_path = frames_in[i-1]
            frame2_path = frames_in[i]
    
            frame1 = PIL.Image.open(frame1_path)
            frame2 = PIL.Image.open(frame2_path)
          
            frame = PIL.Image.fromarray((np.array(frame1)*(1-blend) + np.array(frame2)*(blend)).astype('uint8')).save(batchFolder+f"/blend/{folder}({run})_{i:04}.png")


    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec',
        'png',
        '-r',
        str(fps),
        '-start_number',
        str(init_frame),
        '-i',
        image_path,
        '-frames:v',
        str(last_frame+1),
        '-c:v',
        'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt',
        'yuv420p',
        '-crf',
        '17',
        '-preset',
        'veryslow',
        filepath
    ]

    process = subprocess.Popen(cmd, cwd=f'{batchFolder}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
    else:
        print("The video is ready and saved to the images folder")

    # if view_video_in_cell:
    #     mp4 = open(filepath,'rb').read()
    #     data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    #     display.HTML(f'<video width=400 controls><source src="{data_url}" type="video/mp4"></video>')

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "GnpnrLfMV2jU"
# !! }}
#@title Clone Real-ESRGAN 
%cd /content/
#!rm -rf /content/Real-ESRGAN
!git clone https://github.com/xinntao/Real-ESRGAN.git
%cd Real-ESRGAN
# Set up the environment
# Install basicsr - https://github.com/xinntao/BasicSR
!pip install basicsr==1.3.5
!pip install facexlib
!pip install gfpgan
!pip install -r requirements.txt
!python3 setup.py develop
# Download the pre-trained model
!wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "u1tiyMZJW5td"
# !! }}
#@title Exec Real-ESRGAN
import os
from google.colab import files
import shutil
output_dir =  f"{outDirPath}/{batch_name}"
upload_folder = 'uploads'
result_folder = 'results'
if os.path.isdir(upload_folder):
    shutil.rmtree(upload_folder)
if os.path.isdir(result_folder):
    shutil.rmtree(result_folder)
os.mkdir(upload_folder)
os.mkdir(result_folder)
!cp  {output_dir}/*.png {upload_folder}

# if it is out of memory, try to use the `--tile` option
# We upsample the image with the scale factor X3.5
# RealESRGAN_x4plus
!python inference_realesrgan.py -n RealESRGAN_x4plus -i uploads --outscale 4  --face_enhance
# Arguments
# -n, --model_name: Model names
# -i, --input: input folder or image
# --outscale: Output scale, can be arbitrary scale factore. 

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "oFEI9xJYJaW1"
# !! }}
# @title ### **Create video from upscaled**
#@markdown Video file will save in the same folder as your images.

skip_video_for_run_all = False #@param {type: 'boolean'}

if skip_video_for_run_all == True:
  print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')

else:
  # import subprocess in case this cell is run without the above cells
  import subprocess
  from base64 import b64encode

  latest_run = batchNum

  folder = batch_name #@param
  run = latest_run #@param
  final_frame = 'final_frame'


  init_frame = 0#@param {type:"number"} This is the frame where the video will start
  last_frame = final_frame#@param {type:"number"} You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.
  fps = 12#@param {type:"number"}
  # view_video_in_cell = True #@param {type: 'boolean'}

  frames = []
  # tqdm.write('Generating video...')
  scaledPath = '/content/Real-ESRGAN/results'
  if last_frame == 'final_frame':
    last_frame = len(glob(batchFolder+f"/{folder}({run})_*.png"))
    print(f'Total frames: {last_frame}')

  image_path = f"{scaledPath}/{folder}({run})_%04d_out.png"
  filepath = f"{outDirPath}/{folder}/{folder}({run})_esrgan.mp4"


  cmd = [
      'ffmpeg',
      '-y',
      '-vcodec',
      'png',
      '-r',
      str(fps),
      '-start_number',
      str(init_frame),
      '-i',
      image_path,
      '-frames:v',
      str(last_frame+1),
      '-c:v',
      'libx264',
      '-vf',
      f'fps={fps}',
      '-pix_fmt',
      'yuv420p',
      '-crf',
      '27',
      '-preset',
      'veryslow',
      filepath
  ]

  process = subprocess.Popen(cmd, cwd=f'{batchFolder}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()
  if process.returncode != 0:
      print(stderr)
      raise RuntimeError(stderr)
  else:
      print("The video is ready and saved to the images folder")

  # if view_video_in_cell:
  #     mp4 = open(filepath,'rb').read()
  #     data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  #     display.HTML(f'<video width=400 controls><source src="{data_url}" type="video/mp4"></video>')

# %%
# !! {"metadata":{
# !!   "id": "cFBjVRtR1srY"
# !! }}
#@title Download RIFE and pre-trained model
#@markdown Only needed once for initial setup to download this pacakge to your Google Drive.
!git clone https://github.com/hzwer/arXiv2020-RIFE /content/arXiv2020-RIFE > /dev/null
!pip install --upgrade torch torchvision torchdata torchtext --upgrade
!pip install pytorch-lightning
import torch
import torch.utils.data.backward_compatibility
import torch.utils.data.datapipes as dp
import torch.utils.data.graph
import torch.utils.data.graph_settings
from torch.utils.data import (
    DataLoader,
    DataChunk,
    IterDataPipe,
    MapDataPipe,
    RandomSampler,
    argument_validation,
    runtime_validation,
    runtime_validation_disabled,
    datapipes
)
from torch.utils.data.graph import traverse
#from torch.utils.data.datapipes.utils.common import StreamWrapper
import codecs
import functools
import inspect
import os

import torchtext
import pytorch_lightning as pl

!pip install Vocab 
os.makedirs(f'{model_path}/arXiv2020-RIFE/train_log',exist_ok=True)
#!gdown --id 1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_
#!7z e RIFE_trained_model_v3.6.zip
# "RIFE-3.6": "1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_",
#!gdown --id 1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_ -O  '{model_path}/arXiv2020-RIFE/RIFE_trained_model_v3.6.zip'
# "RIFE-3.8": "1O5KfS3KzZCY3imeCr2LCsntLhutKuAqj",
#!gdown --id 1O5KfS3KzZCY3imeCr2LCsntLhutKuAqj -O  '{model_path}/arXiv2020-RIFE/RIFE_trained_model_v3.8.zip'
# "RIFE-3.9": "1iosmPTt2ayAdSMqnI1cxO_R1-Qhrranp",
#!gdown --id 1iosmPTt2ayAdSMqnI1cxO_R1-Qhrranp -O  '{model_path}/arXiv2020-RIFE/RIFE_trained_model_v3.9.zip'
# "RIFE-4.0": "1mUK9iON6Es14oK46-cCflRoPTeGiI_A9",
#!gdown --id 1mUK9iON6Es14oK46-cCflRoPTeGiI_A9 -O  '{model_path}/arXiv2020-RIFE/RIFE_trained_model_v4.0.zip'
#3.8
!gdown --id 1O5KfS3KzZCY3imeCr2LCsntLhutKuAqj -O  '{model_path}/arXiv2020-RIFE/RIFE_trained_model_v3.8.zip'
#!curl -L -o '{model_path}/arXiv2020-RIFE/RIFE_trained_model_v3.8.zip' -C - 'https://drive.google.com/u/0/uc?id=1O5KfS3KzZCY3imeCr2LCsntLhutKuAqj&export=download' && \
!unzip '{model_path}/arXiv2020-RIFE/RIFE_trained_model_v3.8.zip' -d {model_path}/arXiv2020-RIFE  
!pip install git+https://github.com/rkhamilton/vqgan-clip-generator.git   > /dev/null
!pip install sk-video                                                     > /dev/null
!pip install opencv-python                                                > /dev/null
!pip install moviepy   basicsr   realesrgan > /dev/null
!pip install piexif                                                       > /dev/null
from vqgan_clip import generate, video_tools, esrgan
from vqgan_clip.engine import VQGAN_CLIP_Config
import os
import subprocess
from vqgan_clip import _functional as VF

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "S_6C67O6KJoV"
# !! }}
# @title Create upscaled video for social media
#audin='/content/bouncebouncebounce_wfreestyle.mp3'#@param{type:'string'}
#vidin = f"{outDirPath}/{folder}/{folder}({run})_esrgan.mp4"
vidin = f'{filepath}'#-slomo.mp4'
vidout_audio = f"{outDirPath}/{folder}/{folder}({run})_w_audio.mp4"
ffworkdir = f"{outDirPath}/{folder}/"
audin=init_audio#@param{type:'string'}
%env ffworkdir={ffworkdir}
from pathlib import Path
base=Path(vidin).stem
outbase=Path(vidout_audio).stem
%env base={base}
%env vidin={vidin}
%env audin={audin}
%env vidout_audio={vidout_audio}
!base=`basename "$vidin" .mp4`
# -vf scale=3840:2160
!ffmpeg -y -r {fps} -i  '{audin}'   -i  '{vidin}' -c:v libx264 -b:v 40m -profile main -pix_fmt yuv420p '{vidout_audio}'
video2sync = vidout_audio
#@markdown Use optical flow interpolation with RIFE
interpolation_with_RIFE=False #@param{type:'boolean'}
interpolation_factor=4 #@param{type:'number'}
if interpolation_with_RIFE:
  print(f'Interpolating video...')
  output_RIFE=f'{os.path.splitext(vidout_audio)[0]}_RIFE.mp4'
  video_tools.RIFE_interpolation(input=vidout_audio,
                      output=output_RIFE,
                      interpolation_factor=interpolation_factor )
  print(f'Video with interpolation saved as:\n{output_RIFE}')
  video2sync = output_RIFE

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "waXGnWqqWxh6"
# !! }}
#@title Sync audio and video using ffmpeg 
#video2sync = f"{vidout_audio}" #@param {type:"string"}
vidout_path = f"{outDirPath}/{folder}/{folder}({run})_synced.mp4"
vtempfile =   f"{outDirPath}/{folder}/{folder}({run})_temp.mp4"
%env videofile={video2sync}
%env audiofile={init_audio}
%env vtempfile={vtempfile}
%env vidout_path={vidout_path}
%shell set -x ;\
ffmpeg -y -i "$videofile" -i "$audiofile" -c:v copy -c:a copy "$vtempfile" ;\
vtime=`ffprobe -i $videofile -show_entries format=duration -v quiet -of csv="p=0"` ;\
atime=`ffprobe -i $audiofile -show_entries format=duration -v quiet -of csv="p=0"` ;\
ffmpeg -y -i  "$vtempfile" -vf 'setpts=('"${atime}/${vtime}"')*PTS' "$vidout_path"

# %%
# !! {"metadata":{
# !!   "cellView": "form",
# !!   "id": "rUeiCEiQdPrn"
# !! }}
# @title Create upscaled 360 video for social media
# @title Create upscaled video for social media
#audin='/content/bouncebouncebounce_wfreestyle.mp3'#@param{type:'string'}
#vidin = f"{outDirPath}/{folder}/{folder}({run})_esrgan.mp4"
vidin =  vidout_audio = f"{vidout_path}"
output_4ch_audio=f'{os.path.splitext(vidout_audio)[0]}_4ch.mp4'
output_360=f'{os.path.splitext(vidout_audio)[0]}_360.mp4'
ffworkdir = f"{outDirPath}/{folder}/"
audin=init_audio
outputfps=12#@param{type:'number'}

%env ffworkdir={ffworkdir}
from pathlib import Path
base=Path(vidin).stem
outbase=Path(vidout_audio).stem
%env base={base}
%env vidin={vidin}
%env audin={audin}
%env vidout_audio={vidout_audio}

!base=`basename "$vidin" .mp4`
# -vf scale=3840:2160
inputfps=fps*interpolation_factor
!ffmpeg -y -r {inputfps} -i  '{vidin}' -c:v libx264 -b:v 40m -r {outputfps} -profile main -pix_fmt yuv420p '{base}.encodedforjump.video.360.mono.mp4'	   
!ffmpeg -y -i  '{audin}' -channel_layout 4.0 -c:a aac -b:a 128k -strict -2 '{base}-ACN-SN3D-4ch-aac128.mp4'
!ffmpeg -y -i  '{vidin}' -c:v libx264 -b:v 40m   -r {outputfps} -profile main -pix_fmt yuv420p '{vidout_audio}'
 
%cd /content/
!pip install git+https://github.com/google/spatial-media.git
!git clone https://github.com/google/spatial-media.git
%cd spatial-media
!python spatialmedia -i --stereo=none --spatial-audio '{vidout_audio}' '{output_360}'

# %%
# !! {"main_metadata":{
# !!   "accelerator": "GPU",
# !!   "anaconda-cloud": {},
# !!   "colab": {
# !!     "collapsed_sections": [
# !!       "CreditsChTop",
# !!       "TutorialTop",
# !!       "CheckGPU",
# !!       "InstallDeps",
# !!       "DefMidasFns",
# !!       "DefFns",
# !!       "DefSecModel",
# !!       "DefSuperRes",
# !!       "AnimSetTop",
# !!       "ExtraSetTop",
# !!       "InstallRAFT",
# !!       "CustModel",
# !!       "FlowFns1",
# !!       "FlowFns2"
# !!     ],
# !!     "include_colab_link": true,
# !!     "machine_shape": "hm",
# !!     "name": "AudioRider Diffusion v5.6 [Now with portrait_generator_v001]",
# !!     "private_outputs": true,
# !!     "provenance": []
# !!   },
# !!   "kernelspec": {
# !!     "display_name": "Python 3",
# !!     "language": "python",
# !!     "name": "python3"
# !!   },
# !!   "language_info": {
# !!     "codemirror_mode": {
# !!       "name": "ipython",
# !!       "version": 3
# !!     },
# !!     "file_extension": ".py",
# !!     "mimetype": "text/x-python",
# !!     "name": "python",
# !!     "nbconvert_exporter": "python",
# !!     "pygments_lexer": "ipython3",
# !!     "version": "3.6.1"
# !!   }
# !! }}
