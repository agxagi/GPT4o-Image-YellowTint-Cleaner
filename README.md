# GPT4o-Image-YellowTint-Cleaner
A 1 click Yellow Tint remover for GPT4o images.
It automatically processes images to remove yellow tints while preserving image quality. This utility uses advanced image processing techniques including channel manipulation, histogram equalization, and color correction to restore natural-looking colors in yellowed images.

## Before/After Comparison

| Before | After |
|--------|--------|
| ![Before](https://github.com/user-attachments/assets/531698d7-eece-476a-a173-38d18ae0a379) | ![After](https://github.com/user-attachments/assets/96642a1c-2668-4d1f-b297-e6208fd1b506) | 

| ![Before](https://github.com/user-attachments/assets/53d1c2b6-c224-4321-8fc6-7651cc615320) | ![After](https://github.com/user-attachments/assets/a33fdae0-6816-4e43-84a4-4775200fa725) |

## Features

- **Automatic folder management**: Creates input and output directories on first run
- **Batch processing**: Processes all images in the input folder with a single execution
- **Multiple format support**: Works with JPG, PNG, BMP, TIFF, GIF, and WEBP images
- **Non-destructive workflow**: Original images remain untouched while processed versions are saved to a separate output folder
- **Advanced color correction**: Uses sophisticated channel manipulation and normalization techniques to remove yellow color casts

## Installation

### Prerequisites

- Python 3.6 or higher
- PIL/Pillow
- NumPy

### Setup

1. Clone this repository:
```bash
git clone https://github.com/agxagi/GPT4o-Image-YellowTint-Cleaner.git
cd GPT4o-Image-YellowTint-Cleaner
```

2. Install the required dependencies:
```bash
pip install pillow numpy
```

## Usage

1. Run the script to create the required folders:
```bash
python YellowTint_remover_v1.py
```

2. Place your yellow-tinted images in the `input_images` folder that was automatically created

3. Run the script again to process all images:
```bash
python YellowTint_remover_v1.py
```

4. Find your processed images in the `output` folder

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
