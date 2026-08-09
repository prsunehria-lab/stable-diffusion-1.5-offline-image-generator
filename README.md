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

### 2. Install Python Dependencies
Ensure you have Python 3.10+ installed. Create a virtual environment and install the requirements:
bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### 3. Download the Required AI Models
Create a `models/` directory in the root of your project, and download the following models from Hugging Face. Your folder structure must look exactly like this:

📁 `models/`
  * 📁 `stable-diffusion-v1-5/` 
    * *Download the Diffusers format of SD 1.5 (including unet, vae, text_encoder, etc.) from [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).*
  * 📁 `controlnet-openpose/`
    * *Download from [lllyasviel/control_v11p_sd15_openpose](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/tree/main).*
  * 📁 `controlnet-recolor/` (Optional)
    * *Download from [lllyasviel/control_v11b_sd15_recolor](https://huggingface.co/lllyasviel/control_v11b_sd15_recolor/tree/main).*
  * 📁 `annotators/`
    * *Required for OpenPose detection. Download the DWPose or standard OpenPose annotator models.*

---

## 🖥️ How to Run

1. Ensure your virtual environment is activated.
2. Start the Flask server:
bash
python app.py
3. Open your web browser and navigate to:
**`http://127.0.0.1:5000`**

## 💡 Usage Tips

* **Inpainting:** Select "Modify Area" from the dropdown, upload an image, and use your mouse to draw a yellow mask over the area you want to change. Describe the *new* object in the Positive Prompt.
* **Resolutions:** The app supports standard (512x512) up to extreme high-resolution (1024x1024). Note that generating above 512x512 will automatically trigger a Lanczos upscale and secondary refinement pass for maximum quality.
* **Batch Size:** Generating 4 images at once will require significantly more VRAM. If your application crashes, reduce the batch size to 1.