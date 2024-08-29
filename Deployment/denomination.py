import pytesseract
import PIL.Image
import cv2 as cv
import numpy as np

def extract_roi(image):
    x1, y1 = 536, 195
    x2, y2 = 625, 255
    return image[y1:y2, x1:x2]

#This is a function defined to rescale the images

def rescaleFrame(frame):
    width = int(700)
    height = int(300)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def preprocess_image(image):
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply a less aggressive denoising algorithm (Bilateral Filtering)
    blurred = cv.bilateralFilter(gray, 5, 50, 50)

    # Apply thresholding to binarize the image
    _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return binary

def extract(path):
    # Load the image using OpenCV
    image = cv.imread(path)
    rescaled_image = rescaleFrame(image)
    roi = extract_roi(rescaled_image)
    
    # Preprocess the ROI
    preprocessed_roi = preprocess_image(roi)
    
    # Convert the ROI to a PIL Image for pytesseract to process
    pil_image = PIL.Image.fromarray(preprocessed_roi)
    
    # Use pytesseract to extract text from the ROI
    myconfig = '--psm 13 --oem 3'  # Single line of text
    ans = pytesseract.image_to_string(pil_image, config=myconfig)

    print(f"Extracted Text: {ans}")

    return ans