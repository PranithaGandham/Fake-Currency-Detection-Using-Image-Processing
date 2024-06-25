# FAKE-CURRENCY-DETECTION-USING-IMAGE-PROCESSING
Detect counterfeit currency using image processing. Analyzes features like transparent strips, security strips, serial number &amp; gandhiji print, comparing real vs. fake notes to determine authenticity.

## Overview

This project aims to detect counterfeit currency notes using image processing techniques. It loads images of real and fake currency notes, applies various image processing operations such as thresholding and morphological operations, and then analyzes features to determine the authenticity of the currency.

## Description

The project utilizes computer vision libraries like OpenCV and NumPy to process the images of currency notes. It extracts specific regions of interest, such as the transparent strip and certain patterns, and compares them between real and fake notes. The intensity is calculated to assess the similarity between corresponding regions.

## Files

- **Deployment Folder**: This folder contains a static folder containing the javascript and css files of the website. App.py that contains the python flask code used to integrate the project and the website. Realfiveh.png,Realh.jpg and Realtwoh.jpg are used as a refence to compare the uploaded images by the user.


- **Project.py**: The Python code for the fake currency detection project.

- **denomination.py**: This python file consists a code to extract the denomination of the uploaded code and sets the appropriate image for the co,parision. 

- **app.py**: This flask code integrates the html css java script files with the backend project.py

## Usage

1. **Run app.py**: Open and run the `app.py` notebook using IDE or any compatible environment and open the live server.

2. **Load Images**: In the webpage redirect to the detector section and upload the image and submit it.

3. **Interpret Results**: The interpreted reseult would pop up on the website.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Flask
- Tesseract
- HTML
- CSS
- Java Script

## Contribution

Contributions to the project are welcome! Feel free to fork the repository, make improvements, and submit pull requests