import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
import glob
import random

def rescaleFrame(frame, width, height):
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def grayScale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def extract_roi(image, coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    return image[y1:y2, x1:x2]

def augment_image(image):
    rows, cols = image.shape[:2]
    
    # Random Rotation
    angle = random.uniform(-10, 10)
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv.warpAffine(image, M, (cols, rows))
    
    # Random Translation
    tx = random.uniform(-10, 10)
    ty = random.uniform(-10, 10)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv.warpAffine(image, M, (cols, rows))
    
    # Random Scaling
    scale = random.uniform(0.9, 1.1)
    image = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
    
    return image

def compare_histograms(hist1, hist2, method='kldiv'):
    if method == 'bhattacharyya':
        return cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    elif method == 'kldiv':
        return cv.compareHist(hist1,hist2,cv.HISTCMP_KL_DIV)

def compute_histograms(rois):
    histograms = []
    for roi in rois:
        hist = cv.calcHist([roi], [0], None, [256], [0, 256])
        hist = cv.normalize(hist, hist).flatten()
        histograms.append(hist)
    return histograms

def compare_ssim(img1, img2):
    return ssim(img1, img2)

def load_training_data(denomination):
    training_images = []
    training_histograms = []

    training_path = f"Deployment/DataSet/{denomination}/"
    training_files = glob.glob(training_path + '*.jpg')

    for file in training_files:
        image = cv.imread(file)
        if image is None:
            continue
        
        for _ in range(5):  # Apply data augmentation 5 times per image
            augmented_image = augment_image(image)
            rescaled_image = rescaleFrame(augmented_image, 700, 300)
            gray_image = grayScale(rescaled_image)
            blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)
            equalized_image = cv.equalizeHist(blurred_image)
            edge_detected_image = cv.Canny(equalized_image, 150, 255)

            rois = [
                extract_roi(edge_detected_image, [(0, 56), (32, 150)]),
                extract_roi(edge_detected_image, [(380, 0), (410, 300)]),
                extract_roi(edge_detected_image, [(445, 240), (630, 300)]),
                extract_roi(edge_detected_image, [(152, 65), (385, 300)])
            ]
            
            histograms = compute_histograms(rois)
            training_histograms.append(histograms)

            training_images.append(rois)

    return training_images, training_histograms

def validate(string, denomination) -> str:
    check = cv.imread(string)
    if check is None:
        return "Uploaded image not found or could not be read."

    rescaled_image_check = rescaleFrame(check, 700, 300)
    grayImg_check = grayScale(rescaled_image_check)
    blurred_test = cv.GaussianBlur(grayImg_check, (5, 5), 0)
    equalized_test = cv.equalizeHist(blurred_test)
    test_img = cv.Canny(equalized_test, 150, 255)

    security_mark_coords = [(0, 56), (32, 150)]
    green_strip_coords = [(380, 0), (410, 300)]
    serial_number_coords = [(445, 240), (630, 300)]
    gandhiji_coords = [(152, 65), (385, 300)]

    test_rois = [
        extract_roi(test_img, security_mark_coords),
        extract_roi(test_img, green_strip_coords),
        extract_roi(test_img, serial_number_coords),
        extract_roi(test_img, gandhiji_coords)
    ]

    test_histograms = compute_histograms(test_rois)

    training_images, training_histograms = load_training_data(denomination)

    hist_comparison_results = []
    ssim_comparison_results = []

    print("Histogram comparison results:")
    for training_histogram in training_histograms:
        for test_hist, train_hist in zip(test_histograms, training_histogram):
            hist_comparison_result = compare_histograms(test_hist, train_hist, method='kldiv')
            hist_comparison_results.append(hist_comparison_result)
            # print(f"Test Hist: {test_hist[:5]}, Train Hist: {train_hist[:5]}, Result: {hist_comparison_result}")

    print("SSIM comparison results:")
    for training_rois in training_images:
        for test_roi, train_roi in zip(test_rois, training_rois):
            ssim_comparison_result = compare_ssim(test_roi, train_roi)
            ssim_comparison_results.append(ssim_comparison_result)
            # print(f"SSIM Test ROI: {test_roi.shape}, Train ROI: {train_roi.shape}, Result: {ssim_comparison_result}")

    combined_results = []
    print(f"Histogram results length: {len(hist_comparison_results)}, SSIM results length: {len(ssim_comparison_results)}")
    for hist_result, ssim_result in zip(hist_comparison_results, ssim_comparison_results):
        combined_result = (0.7 * (1 - hist_result)) + (0.3 * ssim_result)
        combined_results.append(combined_result)

    # print("Combined Comparison Results (Histogram + SSIM):")
    # for i, result in enumerate(combined_results):
    #     print(f"ROI {i+1}: {result:.4f}")

    threshold = 0.72
    pass_count = 0
    for i, result in enumerate(combined_results):
        if result > threshold:
            print(f"ROI {i+1} passes with result {result:.4f}")
            pass_count += 1
        else:
            print(f"ROI {i+1} fails with result {result:.4f}")

    is_genuine = pass_count >=180
    print(pass_count)

    return f"{'The note is genuine' if is_genuine else 'It is a fake note'}"
