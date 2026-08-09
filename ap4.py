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
# Ensure static folder points to your specific D: drive project path
app = Flask(__name__, static_folder=r"D:\proj\static")

# Configuration for paths on your system
OUTPUT_DIR = r"D:\proj\static\output"
MODEL_DIR = r"D:\proj\models\stable-diffusion-v1-5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- PIPELINE INITIALIZATION -----------------
# Optimized to fit the base models, ControlNet, and scanners into the 6GB VRAM of the RTX 4050
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

# 2. Load the ControlNet Anatomy Enforcer
print("📦 Loading ControlNet OpenPose Module...")
controlnet = ControlNetModel.from_pretrained(
    r"D:\proj\models\controlnet-openpose",  
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True                   
)

# 3. Create Shared Pipelines to save VRAM
# All three pipelines share the UNet and VAE components from the base model
img2img_pipe = StableDiffusionImg2ImgPipeline(**pipe.components)
cnet_pipe = StableDiffusionControlNetPipeline(**pipe.components, controlnet=controlnet)

# Apply Extreme Low-VRAM Optimizations
# Required to avoid memory crashes on consumer hardware
for p in [pipe, img2img_pipe, cnet_pipe]:
    p.unet.to(memory_format=torch.channels_last)
    p.to("cuda")
    p.vae.enable_slicing()
    p.vae.enable_tiling()

# 4. Load AI Scanners for high-quality detailing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pose_estimator = OpenposeDetector.from_pretrained(
    r"D:\proj\models\annotators",           
    local_files_only=True                   
)
face_detector = MTCNN(keep_all=True, min_face_size=15, thresholds=[0.5, 0.6, 0.6], device=device)

print("✅ Master Pipeline Ready! Hardware locked to RTX 4050.")
# -----------------------------------------------------------

# --- Integrated HTML/JS Front-End Template ---
# Includes new Mode selection and Denoising Strength slider
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
            <p class="text-slate-400">Environment State : <span class="text-yellow-400 font-mono text-xs">MASTER(Txt2Img + Img2Img + ControlNet + ADetailer)</span></p>
        </header>

        <div class="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-xl mb-8">
            <form id="generate-form" class="space-y-5">
                
                <div class="bg-slate-900 p-4 rounded-lg border border-yellow-500/30">
                    <label for="gen_mode" class="block text-sm font-semibold text-yellow-400 mb-2">🤖 Integrated Generation Mode</label>
                    <select id="gen_mode" name="gen_mode" 
                        class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-white focus:outline-none focus:ring-2 focus:ring-yellow-500 transition cursor-pointer">
                        <option value="txt2img" selected>Text to Image (Standard Txt2Img)</option>
                        <option value="img2img">Edit This Image (Direct Img2Img Editing)</option>
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

                <div class="text-xs text-slate-400 bg-slate-900 p-3 rounded-lg border border-slate-700">
                    👁️ <span class="font-semibold text-yellow-400">Smart Engine Active:</span> Resolution upscale, anatomy enforcement, and facial reconstruction run automatically based on your selections.
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
        
        // --- NEW: UI Context Management ---
        const modeSelect = document.getElementById('gen_mode');
        const modeHelpText = document.getElementById('mode_help');
        const strengthContainer = document.getElementById('strength_container');
        const strengthSlider = document.getElementById('img2img_strength');
        const strengthValSpan = document.getElementById('strength_val');

        modeSelect.addEventListener('change', function(e) {
            const mode = e.target.value;
            // 1. Hide strength slider unless in img2img mode
            strengthContainer.classList.add('hidden');
            // 2. Update help text
            if (mode === 'txt2img') {
                modeHelpText.innerText = "Upload an optional image to use with a pose (needs ControlNet mode).";
            } else if (mode === 'img2img') {
                modeHelpText.innerText = "Upload an image. The AI will edit this specific image based on your prompt.";
                strengthContainer.classList.remove('hidden');
            } else if (mode === 'cnet') {
                modeHelpText.innerText = "Upload a photo of a person. The AI extracts their skeleton to define the pose.";
            }
        });

        strengthSlider.addEventListener('input', function(e) {
            strengthValSpan.innerText = e.target.value;
        });

        // --- Core JS Logic Remains/Updated ---
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
            
            // NEW: Get Mode and Strength
            const mode = document.getElementById('gen_mode').value;
            const strength = parseFloat(document.getElementById('img2img_strength').value);
            
            if (!prompt) return alert('Please enter a prompt first.');
            
            // Check mandatory uploads for non-txt2img modes
            if (mode !== 'txt2img' && !poseBase64) {
                return alert('Please upload an input image for Img2Img or ControlNet mode.');
            }

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
                    // NEW: Payload includes mode and strength
                    body: JSON.stringify({ 
                        prompt: prompt,
                        negative_prompt: negativePrompt,
                        width: width,
                        height: height,
                        batch_size: batchSize,
                        mode: mode,
                        strength: strength,
                        pose_image: poseBase64
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    timeSpentSpan.innerText = data.time_taken.toFixed(1);
                    timerBadge.classList.remove('hidden');

                    data.images.forEach(imgUrl => {
                        const wrapper = document.createElement('div');
                        wrapper.className = "bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow-lg transform transition hover:scale-[1.02]";
                        wrapper.innerHTML = `
                            <img src="${imgUrl}" alt="Generated artwork" class="w-full h-auto object-cover bg-slate-950">
                            <div class="p-3 text-xs text-slate-400 bg-slate-850 border-t border-slate-700 truncate">
                                <span class="text-yellow-400 font-mono">[${width}x${height}]</span> ${prompt}
                            </div>
                        `;
                        gallery.insertBefore(wrapper, gallery.firstChild);
                    });
                } else {
                    alert('Error running model: ' + data.error);
                }
            } catch (err) {
                alert('Server communications failed.');
                console.error(err);
            } finally {
                submitBtn.disabled = false;
                submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                loadingArea.classList.add('hidden');
            }
        });
    </script>
</body>
</html>
"""

# --- FEATHERED FACE FIXER ---
# Remains unchanged - crucial for quality
def detail_faces(base_image, prompt, neg_prompt):
    boxes, _ = face_detector.detect(base_image)
    final_image = base_image.copy()
    
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            w, h = x2 - x1, y2 - y1
            pad_w, pad_h = int(w * 0.4), int(h * 0.4) 
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(base_image.width, x2 + pad_w)
            y2 = min(base_image.height, y2 + pad_h)
            
            face_crop = base_image.crop((x1, y1, x2, y2))
            face_512 = face_crop.resize((512, 512), Image.Resampling.LANCZOS)
            
            with torch.inference_mode():
                # Share components via shared pipe
                fixed_face_512 = img2img_pipe(
                    prompt="highly detailed beautiful face, perfect eyes, symmetrical, looking straight, " + prompt, 
                    negative_prompt=neg_prompt + ", ugly, deformed, blurry, bad anatomy",
                    image=face_512,
                    strength=0.30,  
                    num_inference_steps=20,     
                    guidance_scale=7.5
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

# ----------------- MAIN INTEGRATED GENERATION ROUTE -----------------
@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        neg_prompt = data.get('negative_prompt', '').strip()
        target_width = int(data.get('width', 512))
        target_height = int(data.get('height', 512))
        batch_size = int(data.get('batch_size', 4))
        pose_b64 = data.get('pose_image', None)
        
        # NEW INPUT: Mode and Strength
        mode = data.get('mode', 'txt2img')
        img2img_strength = float(data.get('strength', 0.35))
        
        # Guardrail rule to cap batch size safely
        batch_size = max(1, min(batch_size, 4))
        
        if not prompt:
            return jsonify({'success': False, 'error': 'Empty prompt'}), 400

        prompts = [prompt] * batch_size
        neg_prompts = [neg_prompt] * batch_size if neg_prompt else None

        timestamp = int(time.time())
        saved_urls = []
        start_time = time.time()

        # [UPDATED LOGIC] Unified Image Processing based on Mode
        raw_uploaded_img = None
        base_user_image = None
        extracted_pose = None
        
        # Determine how to treat the input image
        if pose_b64:
            image_data = base64.b64decode(pose_b64.split(",")[1])
            raw_uploaded_img = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            if mode == 'img2img':
                # Direct Image Editing Mode: Content is key
                # Resize user's content to standard 512x512 processing size
                base_user_image = raw_uploaded_img.resize((512, 512), Image.Resampling.LANCZOS)
                
            elif mode == 'cnet':
                # Anatomy Enforcer Mode: Skeleton is key
                extracted_pose = pose_estimator(raw_uploaded_img)
            
            # (If mode is 'txt2img', raw_uploaded_img is not used directly)

        with torch.inference_mode():
            # PASS 1: Base Composition - Dynamic pathway selection
            
            if mode == 'img2img' and base_user_image:
                # Modifies the uploaded base_user_image directly
                # High-VRAM optimization via batching the single base image
                base_images = img2img_pipe(
                    prompt=prompts, negative_prompt=neg_prompts,
                    image=[base_user_image] * batch_size,
                    strength=img2img_strength, # New user controlled strength
                    num_inference_steps=25, guidance_scale=7.5
                ).images
                
            elif extracted_pose:
                # Anatomy enforced generation from new content (existing logic)
                base_images = cnet_pipe(
                    prompt=prompts, negative_prompt=neg_prompts,
                    image=extracted_pose,
                    height=512, width=512,
                    num_inference_steps=25, guidance_scale=7.5
                ).images
                
            else:
                # Text-to-Image base logic or FALLBACK (existing logic)
                base_images = pipe(
                    prompt=prompts, negative_prompt=neg_prompts,
                    height=512, width=512,
                    num_inference_steps=25, guidance_scale=7.5
                ).images
            
            # PASS 2: Existing High-Res Detailing Pathway (unchanged)
            outputs = []
            if target_width == 512 and target_height == 512:
                # Square Mode: Just fix faces
                for base_img in base_images:
                    final_img = detail_faces(base_img, prompt, neg_prompt)
                    outputs.append(final_img)
            else:
                # High-Res Mode: Upscale, Refine, and Face Fix
                for base_img in base_images:
                    upscaled_base = base_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    
                    # Refinement strength remains fixed for quality consistency
                    refined_img = img2img_pipe(
                        prompt=prompt, negative_prompt=neg_prompts,
                        image=upscaled_base,
                        strength=0.35,             
                        num_inference_steps=20,     
                        guidance_scale=7.5
                    ).images[0]
                    
                    final_perfect_image = detail_faces(refined_img, prompt, neg_prompt)
                    outputs.append(final_perfect_image)

        total_generation_time = time.time() - start_time

        # Save and return files
        for idx, img in enumerate(outputs):
            filename = f"gen_{timestamp}_{idx}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            img.save(filepath)
            saved_urls.append(f"/static/output/{filename}")

        return jsonify({
            'success': True, 
            'images': saved_urls, 
            'time_taken': total_generation_time
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    # Ensure this matches your local activation parameters
    app.run(host="127.0.0.1", port=5000, debug=False)
