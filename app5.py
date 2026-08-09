import os
import time
import torch
import base64
import io
from flask import Flask, request, jsonify, render_template_string
from PIL import Image, ImageFilter, ImageDraw
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from facenet_pytorch import MTCNN
from controlnet_aux import OpenposeDetector

# Initialize Flask app
app = Flask(__name__, static_folder=r"D:\proj\static")

OUTPUT_DIR = r"D:\proj\static\output"
MODEL_DIR = r"D:\proj\models\stable-diffusion-v1-5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- PIPELINE INITIALIZATION -----------------
print("🚀 Initializing Master ControlNet Pipeline...")

# 1. Load the Base Model
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_DIR, 
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    safety_checker=None,       
    local_files_only=True      
)

# 2a. Load the ControlNet Anatomy Enforcer (Pose)
print("📦 Loading ControlNet OpenPose...")
controlnet_pose = ControlNetModel.from_pretrained(
    r"D:\proj\models\controlnet-openpose",  
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True                   
)

# 2b. Load the ControlNet Recolor Engine (NEW)
print("📦 Loading ControlNet Recolor...")
controlnet_recolor = ControlNetModel.from_pretrained(
    r"D:\proj\models\controlnet-recolor",  
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True                   
)

# 3. Create Shared Pipelines to save VRAM
img2img_pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
cnet_pose_pipe = StableDiffusionControlNetPipeline(**pipe.components, controlnet=controlnet_pose)
cnet_recolor_pipe = StableDiffusionControlNetPipeline(**pipe.components, controlnet=controlnet_recolor)

# Apply Extreme Low-VRAM Optimizations
for p in [pipe, img2img_pipe, cnet_pose_pipe, cnet_recolor_pipe]:
    p.unet.to(memory_format=torch.channels_last)
    p.to("cuda")
    p.vae.enable_slicing()
    p.vae.enable_tiling()

# 4. Load AI Scanners
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pose_estimator = OpenposeDetector.from_pretrained(
    r"D:\proj\models\annotators",           
    local_files_only=True                   
)
face_detector = MTCNN(keep_all=True, min_face_size=15, thresholds=[0.5, 0.6, 0.6], device=device)

print("✅ Master Pipeline Ready! Hardware locked to RTX 4050.")
# -----------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Studio Generator</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <style>
        .loader {
            border-top-color: #eab308;
            animation: spinner 1.5s linear infinite;
        }
        @keyframes spinner {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen font-sans">

    <div class="container mx-auto px-4 py-8 max-w-5xl">
        <header class="text-center mb-10">
            <h1 class="text-4xl font-extrabold tracking-tight text-white mb-2">
                IMAGE GENERATION USING NVIDIA BY VISION X
            </h1>
            <p class="text-slate-400">Environment State : <span class="text-yellow-400 font-mono text-xs">MASTER(Txt2Img + Img2Img + Pose + Recolor + ADetailer)</span></p>
        </header>

        <div class="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-xl mb-8">
            <form id="generate-form" class="space-y-5">
                
                <div class="bg-slate-900 p-4 rounded-lg border border-yellow-500/30">
                    <label for="gen_mode" class="block text-sm font-semibold text-yellow-400 mb-2">🤖 Integrated Generation Mode</label>
                    <select id="gen_mode" name="gen_mode" 
                        class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-yellow-500 transition cursor-pointer">
                        <option value="txt2img" selected>Text to Image (Standard Txt2Img)</option>
                        <option value="img2img">Edit This Image (Direct Img2Img Editing)</option>
                        <option value="recolor">Colorize B&W Image (ControlNet Recolor)</option>
                        <option value="cnet">Use as Pose Reference (ControlNet OpenPose)</option>
                    </select>
                </div>

                <div class="bg-slate-900 p-4 rounded-lg border border-slate-700">
                    <label class="block text-sm font-semibold text-yellow-400 mb-2">🖼️ Input Image (Optional for Txt2Img, Required for Others)</label>
                    <p id="mode_help" class="text-xs text-slate-400 mb-3">Upload an image for the AI to process or use as a guide.</p>
                    <input type="file" id="pose_image" accept="image/*" 
                        class="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-yellow-500/10 file:text-yellow-400 hover:file:bg-yellow-500/20 transition cursor-pointer">
                    
                    <div id="pose_preview_container" class="hidden mt-3">
                        <img id="pose_preview" class="h-32 w-auto rounded border border-slate-600 object-cover">
                    </div>

                    <div id="strength_container" class="hidden mt-3 p-3 bg-slate-950/50 rounded-md border border-slate-700">
                        <label for="img2img_strength" class="block text-sm font-medium text-slate-300 mb-1">Alteration Strength: <span id="strength_val" class="font-bold text-yellow-400">0.35</span></label>
                        <p class="text-xs text-slate-500 mb-2">(0.1 = Subtle Edit | 0.9 = Radical Change)</p>
                        <input type="range" id="img2img_strength" name="img2img_strength" min="0.1" max="0.9" step="0.05" value="0.35" class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer">
                    </div>
                </div>

                <div>
                    <label for="prompt" class="block text-sm font-medium text-slate-300 mb-2">Positive Prompt</label>
                    <textarea id="prompt" name="prompt" rows="3" 
                        class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-yellow-500 transition"
                        placeholder="Describe what you want to generate..."></textarea>
                </div>
                
                <div>
                    <label for="negative_prompt" class="block text-sm font-medium text-slate-300 mb-2">Negative Prompt</label>
                    <textarea id="negative_prompt" name="negative_prompt" rows="2" 
                        class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-red-500 transition"
                        placeholder="blurry, deformed, bad anatomy, duplicate people..."></textarea>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label for="resolution" class="block text-sm font-medium text-slate-300 mb-2">Target Resolution Profile</label>
                        <select id="resolution" name="resolution" 
                            class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-yellow-500 transition cursor-pointer">
                            <optgroup label="Standard Native (Fast)">
                                <option value="512x512" selected>512 x 512 (1:1 Standard Square)</option>
                            </optgroup>
                            <optgroup label="Landscape & Cinematic">
                                <option value="768x512">768 x 512 (3:2 Classic Landscape)</option>
                                <option value="896x512">896 x 512 (7:4 Wide Landscape)</option>
                                <option value="1024x576">1024 x 576 (16:9 Cinematic HD Wide)</option>
                            </optgroup>
                            <optgroup label="Portrait & Mobile">
                                <option value="512x768">512 x 768 (2:3 Classic Portrait)</option>
                                <option value="576x1024">576 x 1024 (9:16 Smartphone Screen)</option>
                            </optgroup>
                        </select>
                    </div>
                    
                    <div>
                        <label for="batch_size" class="block text-sm font-medium text-slate-300 mb-2">Images to Generate (Batch Size)</label>
                        <select id="batch_size" name="batch_size" 
                            class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-yellow-500 transition cursor-pointer">
                            <option value="1">1 Image (Fastest)</option>
                            <option value="2">2 Images</option>
                            <option value="3">3 Images</option>
                            <option value="4" selected>4 Images (Full Batch)</option>
                        </select>
                    </div>
                </div>

                <div class="flex justify-between items-center pt-4 border-t border-slate-700">
                    <span class="text-xs text-slate-400 bg-slate-900 px-3 py-1.5 rounded-md border border-slate-700">
                        ⚡ Hardware: RTX 4050 Master Load Optimization Active
                    </span>
                    <button type="submit" id="submit-btn"
                        class="bg-yellow-600 hover:bg-yellow-500 text-white font-semibold px-6 py-2.5 rounded-lg shadow-md transition transform active:scale-95 flex items-center space-x-2">
                        <span>Generate Master Batch</span>
                    </button>
                </div>
            </form>
        </div>

        <div id="loading-area" class="hidden text-center py-12">
            <div class="loader ease-linear rounded-full border-4 border-t-4 border-yellow-500 h-12 w-12 mx-auto mb-4"></div>
            <p class="text-yellow-400 font-medium animate-pulse">Running Neural Rendering Pipeline...</p>
        </div>

        <main>
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-xl font-bold text-slate-300 flex items-center space-x-2">
                    <span>🖼️ Generated Outputs</span>
                </h2>
                <div id="timer-badge" class="hidden text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full px-3 py-1">
                    ⏱️ Last batch generated in <span id="time-spent" class="font-mono">0.0</span>s
                </div>
            </div>
            
            <div id="gallery" class="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <p id="placeholder-text" class="text-slate-500 col-span-2 text-center py-12 border-2 border-dashed border-slate-800 rounded-xl">
                    No images generated in this session yet. Define your mode above to begin.
                </p>
            </div>
        </main>
    </div>

    <script>
        let poseBase64 = null;
        
        const modeSelect = document.getElementById('gen_mode');
        const modeHelpText = document.getElementById('mode_help');
        const strengthContainer = document.getElementById('strength_container');
        const strengthSlider = document.getElementById('img2img_strength');
        const strengthValSpan = document.getElementById('strength_val');

        modeSelect.addEventListener('change', function(e) {
            const mode = e.target.value;
            strengthContainer.classList.add('hidden');
            
            if (mode === 'txt2img') {
                modeHelpText.innerText = "Upload an optional image to use with a pose (needs ControlNet mode).";
            } else if (mode === 'img2img') {
                modeHelpText.innerText = "Upload an image. The AI will edit this specific image based on your prompt.";
                strengthContainer.classList.remove('hidden');
            } else if (mode === 'recolor') {
                modeHelpText.innerText = "Upload a black-and-white photo. The AI will strictly add color based on your prompt without changing the face.";
            } else if (mode === 'cnet') {
                modeHelpText.innerText = "Upload a photo of a person. The AI extracts their skeleton to define the pose.";
            }
        });

        strengthSlider.addEventListener('input', function(e) {
            strengthValSpan.innerText = e.target.value;
        });

        document.getElementById('pose_image').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    poseBase64 = event.target.result;
                    document.getElementById('pose_preview').src = poseBase64;
                    document.getElementById('pose_preview_container').classList.remove('hidden');
                }
                reader.readAsDataURL(file);
            }
        });

        document.getElementById('generate-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const prompt = document.getElementById('prompt').value.trim();
            const negativePrompt = document.getElementById('negative_prompt').value.trim();
            const resolutionVal = document.getElementById('resolution').value;
            const batchSize = parseInt(document.getElementById('batch_size').value);
            const mode = document.getElementById('gen_mode').value;
            const strength = parseFloat(document.getElementById('img2img_strength').value);
            
            if (!prompt) return alert('Please enter a prompt first.');
            if (mode !== 'txt2img' && !poseBase64) return alert('Please upload an input image for this mode.');

            const [width, height] = resolutionVal.split('x').map(Number);
            const submitBtn = document.getElementById('submit-btn');
            const loadingArea = document.getElementById('loading-area');
            const gallery = document.getElementById('gallery');
            const placeholder = document.getElementById('placeholder-text');
            const timerBadge = document.getElementById('timer-badge');
            const timeSpentSpan = document.getElementById('time-spent');

            submitBtn.disabled = true;
            submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
            loadingArea.classList.remove('hidden');
            if (placeholder) placeholder.remove();

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        prompt: prompt, negative_prompt: negativePrompt,
                        width: width, height: height, batch_size: batchSize,
                        mode: mode, strength: strength, pose_image: poseBase64
                    })
                });
                
                const data = await response.json();
                if (data.success) {
                    timeSpentSpan.innerText = data.time_taken.toFixed(1);
                    timerBadge.classList.remove('hidden');
                    data.images.forEach(imgUrl => {
                        const wrapper = document.createElement('div');
                        wrapper.className = "bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow-lg transform transition hover:scale-[1.02]";
                        wrapper.innerHTML = `<img src="${imgUrl}" class="w-full h-auto object-cover bg-slate-950">`;
                        gallery.insertBefore(wrapper, gallery.firstChild);
                    });
                } else { alert('Error: ' + data.error); }
            } catch (err) { alert('Server failed.'); } finally {
                submitBtn.disabled = false;
                submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                loadingArea.classList.add('hidden');
            }
        });
    </script>
</body>
</html>
"""

def detail_faces(base_image, prompt, neg_prompt):
    boxes, _ = face_detector.detect(base_image)
    final_image = base_image.copy()
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            w, h = x2 - x1, y2 - y1
            pad_w, pad_h = int(w * 0.4), int(h * 0.4) 
            x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
            x2, y2 = min(base_image.width, x2 + pad_w), min(base_image.height, y2 + pad_h)
            face_crop = base_image.crop((x1, y1, x2, y2))
            face_512 = face_crop.resize((512, 512), Image.Resampling.LANCZOS)
            with torch.inference_mode():
                fixed_face_512 = img2img_pipe(
                    prompt="highly detailed beautiful face, perfect eyes, symmetrical, looking straight, " + prompt, 
                    negative_prompt=neg_prompt + ", ugly, deformed, blurry, bad anatomy",
                    image=face_512, strength=0.30, num_inference_steps=20, guidance_scale=7.5
                ).images[0]
            fixed_face_shrunk = fixed_face_512.resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
            mask = Image.new("L", fixed_face_shrunk.size, 0)
            draw = ImageDraw.Draw(mask)
            border = int(min(mask.width, mask.height) * 0.20)
            draw.rectangle([border, border, mask.width - border, mask.height - border], fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(border))
            final_image.paste(fixed_face_shrunk, (x1, y1), mask)
    return final_image

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        neg_prompt = data.get('negative_prompt', '').strip()
        target_width, target_height = int(data.get('width', 512)), int(data.get('height', 512))
        batch_size = max(1, min(int(data.get('batch_size', 4)), 4))
        mode = data.get('mode', 'txt2img')
        img2img_strength = float(data.get('strength', 0.35))
        pose_b64 = data.get('pose_image', None)
        
        prompts, neg_prompts = [prompt] * batch_size, [neg_prompt] * batch_size if neg_prompt else None
        timestamp = int(time.time())
        saved_urls = []
        start_time = time.time()

        base_user_image = None
        extracted_pose = None
        
        if pose_b64:
            image_data = base64.b64decode(pose_b64.split(",")[1])
            raw_uploaded_img = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            if mode == 'img2img':
                base_user_image = raw_uploaded_img.resize((512, 512), Image.Resampling.LANCZOS)
            elif mode == 'recolor':
                # Convert the image to pure Grayscale/Luminance for the Recolor model
                grayscale_img = raw_uploaded_img.convert("L").convert("RGB")
                base_user_image = grayscale_img.resize((512, 512), Image.Resampling.LANCZOS)
            elif mode == 'cnet':
                extracted_pose = pose_estimator(raw_uploaded_img)

        with torch.inference_mode():
            if mode == 'img2img' and base_user_image:
                base_images = img2img_pipe(
                    prompt=prompts, negative_prompt=neg_prompts, image=[base_user_image] * batch_size,
                    strength=img2img_strength, num_inference_steps=25, guidance_scale=7.5
                ).images
            elif mode == 'recolor' and base_user_image:
                # Recolor bypasses the "strength" trap by using the grayscale image as a structure map
                base_images = cnet_recolor_pipe(
                    prompt=prompts, negative_prompt=neg_prompts, image=base_user_image,
                    height=512, width=512, num_inference_steps=25, guidance_scale=7.5
                ).images
            elif extracted_pose:
                base_images = cnet_pose_pipe(
                    prompt=prompts, negative_prompt=neg_prompts, image=extracted_pose,
                    height=512, width=512, num_inference_steps=25, guidance_scale=7.5
                ).images
            else:
                base_images = pipe(
                    prompt=prompts, negative_prompt=neg_prompts, height=512, width=512,
                    num_inference_steps=25, guidance_scale=7.5
                ).images
            
            outputs = []
            if target_width == 512 and target_height == 512:
                for base_img in base_images:
                    outputs.append(detail_faces(base_img, prompt, neg_prompt))
            else:
                for base_img in base_images:
                    upscaled = base_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    refined = img2img_pipe(
                        prompt=prompt, negative_prompt=neg_prompts, image=upscaled,
                        strength=0.35, num_inference_steps=20, guidance_scale=7.5
                    ).images[0]
                    outputs.append(detail_faces(refined, prompt, neg_prompt))

        for idx, img in enumerate(outputs):
            filename = f"gen_{timestamp}_{idx}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            img.save(filepath)
            saved_urls.append(f"/static/output/{filename}")

        return jsonify({'success': True, 'images': saved_urls, 'time_taken': time.time() - start_time})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
