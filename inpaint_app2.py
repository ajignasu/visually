import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import pipeline
from diffusers import StableDiffusionInpaintPipeline
import openai
from openai import OpenAI
import requests
from io import BytesIO
import os

# Initialize OpenAI client - you can set your API key here or as an environment variable
# Option 1: Set directly (not recommended for production)
client = OpenAI(api_key="YOUR KEY GOES HERE")

# Option 2: Set from environment variable (recommended)
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load segmentation and inpainting models
segmenter = pipeline("image-segmentation", model="nvidia/segformer-b1-finetuned-cityscapes-1024-1024")

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=dtype
).to(device)

# Function to generate image using OpenAI's DALL-E 3
def generate_image_dalle3(prompt, size="1024x1024", quality="standard", style="vivid"):
    """Generate image using OpenAI's DALL-E 3"""
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1
        )
        
        # Get the image URL
        image_url = response.data[0].url
        
        # Download the image
        image_response = requests.get(image_url)
        image = Image.open(BytesIO(image_response.content))
        
        return image, "Image generated successfully!"
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

# Utility to get labels and masks
def get_labels_and_results(image):
    results = segmenter(image)
    labels = sorted(set([r["label"] for r in results]))
    return labels, results

# Create mask for a selected label
def create_mask(results, selected_label, image_size):
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for r in results:
        if r["label"] == selected_label:
            # Resize the mask to match the original image size
            segment_mask = r["mask"].resize(image_size, resample=Image.NEAREST)
            # Convert to numpy array and apply
            mask_array = np.array(mask)
            segment_array = np.array(segment_mask)
            mask_array = np.maximum(mask_array, segment_array)
            mask = Image.fromarray(mask_array)
    
    # Ensure the mask is binary (0 or 255)
    mask_array = np.array(mask)
    mask_array = (mask_array > 128).astype(np.uint8) * 255
    return Image.fromarray(mask_array)

# Visualize mask for debugging
def show_mask_on_image(image, mask):
    """Overlay mask on image for visualization"""
    img_array = np.array(image.convert("RGBA"))
    mask_array = np.array(mask)
    
    # Create red overlay where mask is white
    overlay = np.zeros_like(img_array)
    overlay[:, :, 0] = 255  # Red channel
    overlay[:, :, 3] = (mask_array > 128).astype(np.uint8) * 128  # Alpha channel
    
    # Composite
    result = Image.alpha_composite(
        Image.fromarray(img_array),
        Image.fromarray(overlay)
    )
    return result.convert("RGB")

# Main function with mask preview
def process_with_mask_preview(image, selected_label):
    """Generate mask and show preview"""
    if image is None or not selected_label:
        return None, None
    
    _, results = get_labels_and_results(image)
    mask = create_mask(results, selected_label, image.size)
    mask_preview = show_mask_on_image(image, mask)
    
    return mask_preview, mask

# Inpainting function
def inpaint_image(image, mask, prompt, num_steps, guidance_scale, strength):
    if image is None or mask is None or not prompt:
        return None
    
    # Resize for stable diffusion (must be divisible by 8)
    target_size = (512, 512)
    
    # Resize image and mask
    image_resized = image.resize(target_size, Image.LANCZOS)
    mask_resized = mask.resize(target_size, Image.NEAREST)
    
    # Ensure mask is proper format (L mode, binary)
    mask_array = np.array(mask_resized)
    mask_binary = (mask_array > 128).astype(np.uint8) * 255
    mask_final = Image.fromarray(mask_binary, mode='L')
    
    # Run inpainting
    with torch.no_grad():
        output = inpaint_pipe(
            prompt=prompt,
            image=image_resized,
            mask_image=mask_final,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            height=512,
            width=512
        ).images[0]
    
    # Resize back to original size
    output_resized = output.resize(image.size, Image.LANCZOS)
    
    return output_resized

# Update label choices dynamically
def update_labels(image):
    if image is None:
        return gr.Dropdown(choices=[], value=None)
    
    labels, _ = get_labels_and_results(image)
    return gr.Dropdown(choices=labels, value=labels[0] if labels else None)

# Combined function for the full pipeline
def full_inpaint_pipeline(image, selected_label, prompt, num_steps, guidance_scale, strength):
    if image is None or not selected_label or not prompt:
        return None, None, None
    
    # Get mask
    _, results = get_labels_and_results(image)
    mask = create_mask(results, selected_label, image.size)
    mask_preview = show_mask_on_image(image, mask)
    
    # Do inpainting
    result = inpaint_image(image, mask, prompt, num_steps, guidance_scale, strength)
    
    return mask_preview, mask, result

# Function to handle tab change and update image
def on_tab_change(tab_index, uploaded_image, generated_image):
    """Update the working image based on selected tab"""
    if tab_index == 0:  # Upload tab
        return uploaded_image
    else:  # Generate tab
        return generated_image

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎨 Text-Guided Inpainting with Semantic Masks")
    gr.Markdown("Generate an image with DALL-E 3 or upload your own, then use AI to intelligently replace objects!")
    
    with gr.Row():
        with gr.Column():
            # Tab interface for image input
            with gr.Tabs() as tabs:
                with gr.TabItem("Upload Image", id=0):
                    uploaded_image = gr.Image(label="Upload an image", type="pil")
                
                with gr.TabItem("Generate Image", id=1):
                    generation_prompt = gr.Textbox(
                        label="Describe the image you want to generate",
                        placeholder="e.g., A modern city street with cars and buildings",
                        lines=2
                    )
                    
                    with gr.Row():
                        size_dropdown = gr.Dropdown(
                            label="Size",
                            choices=["1024x1024", "1024x1792", "1792x1024"],
                            value="1024x1024"
                        )
                        quality_dropdown = gr.Dropdown(
                            label="Quality",
                            choices=["standard", "hd"],
                            value="standard"
                        )
                        style_dropdown = gr.Dropdown(
                            label="Style",
                            choices=["vivid", "natural"],
                            value="vivid"
                        )
                    
                    generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                    generated_image = gr.Image(label="Generated image", type="pil")
                    generation_status = gr.Textbox(label="Status", visible=False)
            
            # Common controls for both upload and generated images
            current_image = gr.Image(label="Current Working Image", type="pil", visible=False)
            
            gr.Markdown("### Inpainting Controls")
            label_dropdown = gr.Dropdown(
                label="Choose object to remove", 
                choices=[],
                interactive=True
            )
            inpaint_prompt = gr.Textbox(
                label="What should replace the object?", 
                placeholder="e.g., lush green grass, beautiful flowers, clear blue sky",
                lines=2
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                num_steps = gr.Slider(
                    minimum=20, 
                    maximum=100, 
                    value=50, 
                    step=5,
                    label="Inference Steps (higher = better quality, slower)"
                )
                guidance_scale = gr.Slider(
                    minimum=1, 
                    maximum=20, 
                    value=7.5, 
                    step=0.5,
                    label="Guidance Scale (higher = follows prompt more closely)"
                )
                strength = gr.Slider(
                    minimum=0.5, 
                    maximum=1.0, 
                    value=1.0, 
                    step=0.05,
                    label="Inpainting Strength (higher = more change)"
                )
            
            inpaint_btn = gr.Button("🖌️ Inpaint Selected Object", variant="primary")
        
        with gr.Column():
            mask_preview = gr.Image(label="Mask Preview (red = area to inpaint)")
            mask_output = gr.Image(label="Generated Mask", visible=False)
            output_image = gr.Image(label="Inpainted Result")
    
    # Event handlers
    
    # Generate image with DALL-E 3
    generate_btn.click(
        fn=generate_image_dalle3,
        inputs=[generation_prompt, size_dropdown, quality_dropdown, style_dropdown],
        outputs=[generated_image, generation_status]
    )
    
    # Update labels when uploaded image changes
    uploaded_image.change(
        fn=update_labels, 
        inputs=uploaded_image, 
        outputs=label_dropdown
    )
    
    # Update labels when generated image changes
    generated_image.change(
        fn=update_labels,
        inputs=generated_image,
        outputs=label_dropdown
    )
    
    # Show mask preview when label is selected for uploaded image
    def update_mask_for_uploaded(image, label):
        if image is not None:
            return process_with_mask_preview(image, label)
        return None, None
    
    # Show mask preview when label is selected for generated image
    def update_mask_for_generated(image, label):
        if image is not None:
            return process_with_mask_preview(image, label)
        return None, None
    
    # State to track current tab
    current_tab = gr.State(value=0)
    
    # Update current tab when changed
    def update_current_tab(evt: gr.SelectData):
        return evt.index
    
    tabs.select(fn=update_current_tab, outputs=[current_tab])
    
    # Determine which image to use for mask preview
    def update_mask_preview(uploaded, generated, label, tab_idx):
        if tab_idx == 0 and uploaded is not None:
            return process_with_mask_preview(uploaded, label)
        elif tab_idx == 1 and generated is not None:
            return process_with_mask_preview(generated, label)
        return None, None
    
    label_dropdown.change(
        fn=update_mask_preview,
        inputs=[uploaded_image, generated_image, label_dropdown, current_tab],
        outputs=[mask_preview, mask_output]
    )
    
    # Inpaint based on current tab
    def inpaint_current_image(uploaded, generated, label, prompt, steps, guidance, strength, tab_idx):
        # Determine which image to use
        if tab_idx == 0:
            current = uploaded
        else:
            current = generated
        
        if current is None:
            return None, None, None
        
        return full_inpaint_pipeline(current, label, prompt, steps, guidance, strength)
    
    inpaint_btn.click(
        fn=inpaint_current_image, 
        inputs=[uploaded_image, generated_image, label_dropdown, inpaint_prompt, 
                num_steps, guidance_scale, strength, current_tab], 
        outputs=[mask_preview, mask_output, output_image]
    )
    
    gr.Markdown("""
    ### Instructions:
    1. **Get an image**: Either upload your own or generate one with DALL-E 3
    2. **Select an object**: Choose from the detected objects in the dropdown
    3. **Describe replacement**: Write what should replace the selected object
    4. **Inpaint**: Click the inpaint button to see the magic!
    
    ### Tips for better results:
    - For DALL-E 3: Be specific in your prompts for better generation results
    - Use descriptive inpainting prompts that match the scene
    - Increase inference steps for better quality (but slower processing)
    - Adjust guidance scale if the result doesn't match your prompt well
    """)

# Add this at the bottom to help users set up their API key

demo.launch()
