import gradio as gr
from PIL import Image, ImageEnhance, ImageChops
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
import time

def normalize_gray(image: Image) -> Image:
    """Normalize a grayscale image using histogram equalization."""
    if image.mode != 'L':
        image = image.convert('L')
    img = np.asarray(image)
    balanced_img = img.copy()
    hist, bins = np.histogram(img.reshape(-1), 256, (0, 256))
    bmin = np.min(np.where(hist > (hist.sum() * 0.0005)))
    bmax = np.max(np.where(hist > (hist.sum() * 0.0005)))
    balanced_img = np.clip(img, bmin, bmax)
    balanced_img = ((balanced_img - bmin) / (bmax - bmin) * 255)
    return Image.fromarray(balanced_img).convert('L')

def image_channel_split(image: Image, mode: str = 'RGBA') -> tuple:
    """Split image into channels based on color mode."""
    _image = image.convert('RGBA')
    channel1 = Image.new('L', size=_image.size, color='black')
    channel2 = Image.new('L', size=_image.size, color='black')
    channel3 = Image.new('L', size=_image.size, color='black')
    channel4 = Image.new('L', size=_image.size, color='black')
    
    if mode == 'RGBA':
        channel1, channel2, channel3, channel4 = _image.split()
    elif mode == 'RGB':
        channel1, channel2, channel3 = _image.convert('RGB').split()
    elif mode == 'YCbCr':
        channel1, channel2, channel3 = _image.convert('YCbCr').split()
    elif mode == 'LAB':
        channel1, channel2, channel3 = _image.convert('LAB').split()
    elif mode == 'HSV':
        channel1, channel2, channel3 = _image.convert('HSV').split()
    
    return channel1, channel2, channel3, channel4

def image_channel_merge(channels: tuple, mode: str = 'RGB') -> Image:
    """Merge channels back into an image based on color mode."""
    channel1 = channels[0].convert('L')
    channel2 = channels[1].convert('L')
    channel3 = channels[2].convert('L')
    channel4 = Image.new('L', size=channel1.size, color='white')
    
    if mode == 'RGBA':
        if len(channels) > 3:
            channel4 = channels[3].convert('L')
        ret_image = Image.merge('RGBA', [channel1, channel2, channel3, channel4])
    elif mode == 'RGB':
        ret_image = Image.merge('RGB', [channel1, channel2, channel3])
    elif mode == 'YCbCr':
        ret_image = Image.merge('YCbCr', [channel1, channel2, channel3]).convert('RGB')
    elif mode == 'LAB':
        ret_image = Image.merge('LAB', [channel1, channel2, channel3]).convert('RGB')
    elif mode == 'HSV':
        ret_image = Image.merge('HSV', [channel1, channel2, channel3]).convert('RGB')
    
    return ret_image

def balance_to_gamma(balance: int) -> float:
    """Convert color balance value to gamma value."""
    return 0.00005 * balance * balance - 0.01 * balance + 1

def gamma_trans(image: Image, gamma: float) -> Image:
    """Apply gamma correction to an image."""
    if gamma == 1.0:
        return image
    img_array = np.array(image)
    img_array = np.power(img_array / 255.0, gamma) * 255.0
    return Image.fromarray(img_array.astype(np.uint8))

def RGB2RGBA(image: Image, mask: Image) -> Image:
    """Convert RGB image to RGBA using provided mask."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if mask.mode != 'L':
        mask = mask.convert('L')
    return Image.merge('RGBA', (*image.split(), mask))

def chop_image_v2(background_image: Image, layer_image: Image, blend_mode: str, opacity: int) -> Image:
    """Blend two images together with specified blend mode and opacity."""
    if background_image.mode != 'RGB':
        background_image = background_image.convert('RGB')
    if layer_image.mode != 'RGB':
        layer_image = layer_image.convert('RGB')
    
    # Convert opacity to float (0-1)
    opacity = opacity / 100.0
    
    # Create a copy of the background image
    result = background_image.copy()
    
    # Apply blend mode
    if blend_mode == "normal":
        result = Image.blend(background_image, layer_image, opacity)
    elif blend_mode == "multiply":
        result = ImageChops.multiply(background_image, layer_image)
        result = Image.blend(background_image, result, opacity)
    elif blend_mode == "screen":
        result = ImageChops.screen(background_image, layer_image)
        result = Image.blend(background_image, result, opacity)
    elif blend_mode == "overlay":
        result = ImageChops.overlay(background_image, layer_image)
        result = Image.blend(background_image, result, opacity)
    
    return result

def auto_adjust(image: Image, strength: int = 100, brightness: int = 0, 
                contrast: int = 0, saturation: int = 0, 
                red: int = 0, green: int = 0, blue: int = 0,
                mode: str = 'RGB') -> Image:
    """
    Apply automatic adjustments to an image.
    
    Args:
        image: PIL Image to adjust
        strength: Overall strength of the adjustment (0-100)
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)
        saturation: Saturation adjustment (-100 to 100)
        red: Red channel adjustment (-100 to 100)
        green: Green channel adjustment (-100 to 100)
        blue: Blue channel adjustment (-100 to 100)
        mode: Color mode for processing ('RGB', 'lum + sat', 'luminance', 'saturation', 'mono')
    
    Returns:
        Adjusted PIL Image
    """
    def auto_level_gray(image):
        """Apply auto levels to a grayscale image."""
        gray_image = Image.new("L", image.size, color='gray')
        gray_image.paste(image.convert('L'))
        return normalize_gray(gray_image)

    # Calculate adjustment factors
    if brightness < 0:
        brightness_offset = brightness / 100 + 1
    else:
        brightness_offset = brightness / 50 + 1
        
    if contrast < 0:
        contrast_offset = contrast / 100 + 1
    else:
        contrast_offset = contrast / 50 + 1
        
    if saturation < 0:
        saturation_offset = saturation / 100 + 1
    else:
        saturation_offset = saturation / 50 + 1

    # Get color channel gammas
    red_gamma = balance_to_gamma(red)
    green_gamma = balance_to_gamma(green)
    blue_gamma = balance_to_gamma(blue)

    # Process image based on mode
    if mode == 'RGB':
        r, g, b, _ = image_channel_split(image, mode='RGB')
        r = auto_level_gray(r)
        g = auto_level_gray(g)
        b = auto_level_gray(b)
        ret_image = image_channel_merge((r, g, b), 'RGB')
    elif mode == 'lum + sat':
        h, s, v, _ = image_channel_split(image, mode='HSV')
        s = auto_level_gray(s)
        ret_image = image_channel_merge((h, s, v), 'HSV')
        l, a, b, _ = image_channel_split(ret_image, mode='LAB')
        l = auto_level_gray(l)
        ret_image = image_channel_merge((l, a, b), 'LAB')
    elif mode == 'luminance':
        l, a, b, _ = image_channel_split(image, mode='LAB')
        l = auto_level_gray(l)
        ret_image = image_channel_merge((l, a, b), 'LAB')
    elif mode == 'saturation':
        h, s, v, _ = image_channel_split(image, mode='HSV')
        s = auto_level_gray(s)
        ret_image = image_channel_merge((h, s, v), 'HSV')
    else:  # mono
        gray = image.convert('L')
        ret_image = auto_level_gray(gray).convert('RGB')

    # Apply color channel adjustments if not in mono mode
    if (red or green or blue) and mode != "mono":
        r, g, b, _ = image_channel_split(ret_image, mode='RGB')
        if red:
            r = gamma_trans(r, red_gamma).convert('L')
        if green:
            g = gamma_trans(g, green_gamma).convert('L')
        if blue:
            b = gamma_trans(b, blue_gamma).convert('L')
        ret_image = image_channel_merge((r, g, b), 'RGB')

    # Apply brightness, contrast, and saturation
    if brightness:
        brightness_image = ImageEnhance.Brightness(ret_image)
        ret_image = brightness_image.enhance(factor=brightness_offset)
        
    if contrast:
        contrast_image = ImageEnhance.Contrast(ret_image)
        ret_image = contrast_image.enhance(factor=contrast_offset)
        
    if saturation:
        color_image = ImageEnhance.Color(ret_image)
        ret_image = color_image.enhance(factor=saturation_offset)

    # Blend with original image based on strength
    ret_image = chop_image_v2(image, ret_image, blend_mode="normal", opacity=strength)
    
    # Handle RGBA mode
    if image.mode == 'RGBA':
        ret_image = RGB2RGBA(ret_image, image.split()[-1])
    
    return ret_image

# Gradio interface functions
def process_single_image(image, strength, brightness, contrast, saturation, red, green, blue, mode):
    """Process a single image with the given parameters."""
    if image is None:
        return None
    
    # Convert from numpy array to PIL Image
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Apply adjustments
    result = auto_adjust(
        pil_image,
        strength=strength,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        red=red,
        green=green,
        blue=blue,
        mode=mode
    )
    
    # Convert back to numpy for Gradio
    return np.array(result)

def process_batch_images(files, strength, brightness, contrast, saturation, red, green, blue, mode):
    """Process multiple uploaded files."""
    if not files:
        return "No files uploaded."
    
    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp()
    output_paths = []
    
    for file_obj in files:
        try:
            # Open image
            input_image = Image.open(file_obj.name)
            
            # Process image
            result = auto_adjust(
                input_image,
                strength=strength,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                red=red,
                green=green,
                blue=blue,
                mode=mode
            )
            
            # Save to temp directory
            output_filename = f"processed_{os.path.basename(file_obj.name)}"
            output_path = os.path.join(temp_dir, output_filename)
            result.save(output_path)
            output_paths.append(output_path)
        except Exception as e:
            print(f"Error processing {file_obj.name}: {e}")
    
    return output_paths

# Create the Gradio app
def create_gradio_interface():
    with gr.Blocks(title="Yellow Tint Cleaner") as app:
        gr.Markdown("# Yellow Tint Cleaner")
        gr.Markdown("Upload an image to remove yellow tints and enhance colors.")
        
        with gr.Tabs():
            with gr.TabItem("Single Image"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Input Image", type="numpy")
                        with gr.Row():
                            gr.Markdown("### Quick Presets")
                        with gr.Row():
                            light_yellow_btn = gr.Button("Light Yellow Tint")
                            medium_yellow_btn = gr.Button("Medium Yellow Tint") 
                            heavy_yellow_btn = gr.Button("Heavy Yellow Tint")
                            old_photo_btn = gr.Button("Old Photo")
                            reset_btn = gr.Button("Reset Parameters")
                        
                        with gr.Row():
                            gr.Markdown("### Adjustment Parameters")
                        
                        with gr.Row():
                            strength = gr.Slider(minimum=0, maximum=100, value=100, step=1, label="Strength")
                        
                        with gr.Row():
                            with gr.Column():
                                brightness = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Brightness")
                                contrast = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Contrast")
                                saturation = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Saturation")
                            
                            with gr.Column():
                                red = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Red")
                                green = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Green")
                                blue = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Blue")
                        
                        mode = gr.Radio(
                            ["RGB", "lum + sat", "luminance", "saturation", "mono"], 
                            value="RGB", 
                            label="Processing Mode"
                        )
                        
                        process_btn = gr.Button("Process Image", variant="primary")
                    
                    with gr.Column():
                        output_image = gr.Image(label="Output Image", type="numpy")
            
            with gr.TabItem("Batch Processing"):
                with gr.Row():
                    with gr.Column():
                        input_files = gr.File(file_count="multiple", label="Upload Images for Batch Processing")
                        
                        with gr.Row():
                            gr.Markdown("### Quick Presets")
                        with gr.Row():
                            batch_light_yellow_btn = gr.Button("Light Yellow Tint")
                            batch_medium_yellow_btn = gr.Button("Medium Yellow Tint") 
                            batch_heavy_yellow_btn = gr.Button("Heavy Yellow Tint")
                            batch_old_photo_btn = gr.Button("Old Photo")
                            batch_reset_btn = gr.Button("Reset Parameters")
                        
                        with gr.Row():
                            gr.Markdown("### Adjustment Parameters")
                        
                        with gr.Row():
                            batch_strength = gr.Slider(minimum=0, maximum=100, value=100, step=1, label="Strength")
                        
                        with gr.Row():
                            with gr.Column():
                                batch_brightness = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Brightness")
                                batch_contrast = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Contrast")
                                batch_saturation = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Saturation")
                            
                            with gr.Column():
                                batch_red = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Red")
                                batch_green = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Green")
                                batch_blue = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Blue")
                        
                        batch_mode = gr.Radio(
                            ["RGB", "lum + sat", "luminance", "saturation", "mono"], 
                            value="RGB", 
                            label="Processing Mode"
                        )
                        
                        batch_process_btn = gr.Button("Process All Images", variant="primary")
                    
                    with gr.Column():
                        output_files = gr.Files(label="Processed Images")
        
        with gr.Accordion("About", open=False):
            gr.Markdown("""
            ## About Yellow Tint Cleaner
            
            This tool uses advanced image processing techniques to remove yellow tints and restore natural colors in images.
            
            ### How to use:
            
            1. Upload an image or multiple images
            2. Adjust the parameters or use one of the presets
            3. Click 'Process Image' or 'Process All Images'
            4. Download the processed results
            
            ### Parameters:
            
            - **Strength**: Controls the overall intensity of the correction
            - **Brightness/Contrast/Saturation**: Basic image adjustments
            - **Red/Green/Blue**: Color channel adjustments (reducing green and increasing blue helps with yellow tints)
            - **Processing Mode**: Different color space approaches for correction
            
            For yellowed photos, try reducing the green channel and increasing the blue channel.
            """)
        
        # Set up events for the single image tab
        process_btn.click(
            process_single_image,
            inputs=[input_image, strength, brightness, contrast, saturation, red, green, blue, mode],
            outputs=output_image
        )
        
        # Set up events for the batch processing tab
        batch_process_btn.click(
            process_batch_images,
            inputs=[input_files, batch_strength, batch_brightness, batch_contrast, batch_saturation, 
                   batch_red, batch_green, batch_blue, batch_mode],
            outputs=output_files
        )
        
        # Define preset functions
        def set_light_yellow_preset():
            return {
                strength: 100,
                brightness: 5,
                contrast: 10,
                saturation: 5,
                red: -5,
                green: -15,
                blue: 20,
                mode: "RGB"
            }
        
        def set_medium_yellow_preset():
            return {
                strength: 100,
                brightness: 10,
                contrast: 15,
                saturation: 10,
                red: -10,
                green: -25,
                blue: 30,
                mode: "RGB"
            }
        
        def set_heavy_yellow_preset():
            return {
                strength: 100,
                brightness: 15,
                contrast: 20,
                saturation: 15,
                red: -15,
                green: -35,
                blue: 40,
                mode: "RGB"
            }
        
        def set_old_photo_preset():
            return {
                strength: 100,
                brightness: 20,
                contrast: 30,
                saturation: 20,
                red: -10,
                green: -20,
                blue: 30,
                mode: "lum + sat"
            }
        
        def reset_parameters():
            return {
                strength: 100,
                brightness: 0,
                contrast: 0,
                saturation: 0,
                red: 0,
                green: 0,
                blue: 0,
                mode: "RGB"
            }
        
        # Set up batch preset functions
        def set_batch_light_yellow_preset():
            return {
                batch_strength: 100,
                batch_brightness: 5,
                batch_contrast: 10,
                batch_saturation: 5,
                batch_red: -5,
                batch_green: -15,
                batch_blue: 20,
                batch_mode: "RGB"
            }
        
        def set_batch_medium_yellow_preset():
            return {
                batch_strength: 100,
                batch_brightness: 10,
                batch_contrast: 15,
                batch_saturation: 10,
                batch_red: -10,
                batch_green: -25,
                batch_blue: 30,
                batch_mode: "RGB"
            }
        
        def set_batch_heavy_yellow_preset():
            return {
                batch_strength: 100,
                batch_brightness: 15,
                batch_contrast: 20,
                batch_saturation: 15,
                batch_red: -15,
                batch_green: -35,
                batch_blue: 40,
                batch_mode: "RGB"
            }
        
        def set_batch_old_photo_preset():
            return {
                batch_strength: 100,
                batch_brightness: 20,
                batch_contrast: 30,
                batch_saturation: 20,
                batch_red: -10,
                batch_green: -20,
                batch_blue: 30,
                batch_mode: "lum + sat"
            }
        
        def reset_batch_parameters():
            return {
                batch_strength: 100,
                batch_brightness: 0,
                batch_contrast: 0,
                batch_saturation: 0,
                batch_red: 0,
                batch_green: 0,
                batch_blue: 0,
                batch_mode: "RGB"
            }
        
        # Connect preset buttons
        light_yellow_btn.click(set_light_yellow_preset, outputs=[strength, brightness, contrast, saturation, red, green, blue, mode])
        medium_yellow_btn.click(set_medium_yellow_preset, outputs=[strength, brightness, contrast, saturation, red, green, blue, mode])
        heavy_yellow_btn.click(set_heavy_yellow_preset, outputs=[strength, brightness, contrast, saturation, red, green, blue, mode])
        old_photo_btn.click(set_old_photo_preset, outputs=[strength, brightness, contrast, saturation, red, green, blue, mode])
        reset_btn.click(reset_parameters, outputs=[strength, brightness, contrast, saturation, red, green, blue, mode])
        
        # Connect batch preset buttons
        batch_light_yellow_btn.click(set_batch_light_yellow_preset, outputs=[batch_strength, batch_brightness, batch_contrast, batch_saturation, batch_red, batch_green, batch_blue, batch_mode])
        batch_medium_yellow_btn.click(set_batch_medium_yellow_preset, outputs=[batch_strength, batch_brightness, batch_contrast, batch_saturation, batch_red, batch_green, batch_blue, batch_mode])
        batch_heavy_yellow_btn.click(set_batch_heavy_yellow_preset, outputs=[batch_strength, batch_brightness, batch_contrast, batch_saturation, batch_red, batch_green, batch_blue, batch_mode])
        batch_old_photo_btn.click(set_batch_old_photo_preset, outputs=[batch_strength, batch_brightness, batch_contrast, batch_saturation, batch_red, batch_green, batch_blue, batch_mode])
        batch_reset_btn.click(reset_batch_parameters, outputs=[batch_strength, batch_brightness, batch_contrast, batch_saturation, batch_red, batch_green, batch_blue, batch_mode])
        
    return app

# Launch the Gradio app
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch()