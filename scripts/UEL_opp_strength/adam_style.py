import os

# Path to Adam's preferred font
FONT_PATH = r'E:\Data\TSDP\helpers\Nexa-ExtraLight.ttf'

# Color palette and styling parameters based on Adam's preference
ADAM_STYLE = {
    'bg_color': '#3c3d3d',
    'acc_color': '#5ECB43', # Brand green
    'text_color': 'white',
    'line_color': '#E74C3C', # Contrast red for regression line
    'scatter_color': '#5888FF', 
    'font_name': 'sans-serif',
    'font_path': FONT_PATH if os.path.exists(FONT_PATH) else None,
    'signature': 'ADAM JAKUS'
}
