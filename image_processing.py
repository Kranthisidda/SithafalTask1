import cv2
import numpy as np

def process_image(file):
    # Read the image file
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Example processing: Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Further processing can be added here
    return {"shape": gray_image.shape, "status": "Processed to grayscale"}