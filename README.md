# GPT4o-Image-YellowTint-remover
A simple & powerful Gradio based webapp to remove yellow tints from an image while preserving the image quality. It uses advanced image processing techniques including channel manipulation, histogram equalization, and color correction to restore natural-looking colors in yellowed images.

## Gradio WebApp Sample
![Gradio](https://github.com/user-attachments/assets/a3272c61-ea64-4c69-9490-4c0c835f4865)

## Best Setting 
![setting](https://github.com/user-attachments/assets/54cbaa2d-c7fe-4283-8670-9dc75a3b6471)

## Before/After Comparison
| Before | After |
|--------|--------|
| ![Before](https://github.com/user-attachments/assets/531698d7-eece-476a-a173-38d18ae0a379) | ![After](https://github.com/user-attachments/assets/e56be65f-abc8-46f2-bed6-6b046c0e0735) | 
| ![Before](https://github.com/user-attachments/assets/40c500a9-8a35-42d2-af59-921bbe9b17ba) | ![After](https://github.com/user-attachments/assets/d8a2817b-4e16-4de6-bbc3-a85aa507249f) |

### Setup

1. Clone this repository:
```bash
git clone https://github.com/agxagi/GPT4o-Image-YellowTint-Cleaner.git
cd GPT4o-Image-YellowTint-Cleaner
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Just run the YellowTint_remover_Gradio.bat file and copy past the URL to your Browser

## How It Works

The Yellow Tint Cleaner uses several image processing techniques to restore natural colors:

1. **Channel Normalization**: Each color channel is normalized individually to correct color imbalances
2. **Histogram Equalization**: Enhances contrast and improves overall image appearance
3. **Color Correction**: Fine-tuned adjustments to remove yellow color casts
4. **Image Blending**: Smooth integration of corrections with the original image

## Customization

You can modify the image processing parameters in the script to adjust the strength of correction:

```python
adjusted_image = auto_adjust(
    input_image,
    strength=100,    # Overall correction strength (0-100)
    brightness=10,   # Brightness adjustment (-100 to 100)
    contrast=15,     # Contrast adjustment (-100 to 100)
    saturation=0,    # Saturation adjustment (-100 to 100)
    red=-5,          # Red channel adjustment (-100 to 100)
    green=-10,       # Green channel adjustment (-100 to 100)
    blue=15,         # Blue channel adjustment (-100 to 100)
    mode='RGB'       # Processing mode
)
```

For yellowed images, try reducing green and red channels while increasing the blue channel.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with PIL/Pillow and NumPy for efficient and effective image restoration
- Inspired by the need to restore old photographs and scanned documents that have developed yellow tints over time
