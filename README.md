# AI Studio: Stable Diffusion 1.5 Offline Generator 🎨🤖

A highly optimized, fully offline web interface for generating and editing images using Stable Diffusion 1.5 and ControlNet. Built with Python, Flask, and TailwindCSS, this application is designed to run entirely locally on NVIDIA GPUs with strict VRAM protections and memory optimizations.

## ✨ Features

* **Multi-Mode Neural Rendering:**
  * **Text-to-Image:** Standard generation with customizable resolutions and aspect ratios.
  * **Image-to-Image:** Direct structural editing with adjustable alteration strength.
  * **Inpainting:** Built-in interactive HTML canvas to draw masks and replace specific objects or areas.
  * **ControlNet OpenPose:** Extract human skeletal architecture from a reference image to guide generation.
  * **ControlNet Recolor:** Seamlessly colorize black-and-white photos without destroying original features.
* **Integrated ADetailer (Face Detailing):** Automatically detects faces using MTCNN and runs a secondary high-resolution pass to fix "messed up" AI faces.
* **VRAM Optimized:** Implements `channels_last` memory formatting, VAE slicing, VAE tiling, and shared pipeline components to prevent out-of-memory crashes on consumer GPUs.

---

## 🚀 Installation & Setup

Because AI models are extremely large, they are not included in this repository. You must download them separately and place them in the correct folders before running the app.

### 1. Clone the Repository
Clone this repository to your local machine (The code defaults to `D:\proj`, but you can adjust the paths in the script if needed):
```bash
git clone [https://github.com/prsunehria-lab/stable-diffusion-1.5-offline-image-generator.git](https://github.com/prsunehria-lab/stable-diffusion-1.5-offline-image-generator.git)
cd stable-diffusion-1.5-offline-image-generator
