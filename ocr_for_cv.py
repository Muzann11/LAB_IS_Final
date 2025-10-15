import cv2
import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"D:\LAB IS\Tesseract-OCR\tesseract.exe"
def ocr_image(image_file):
    image_pil = Image.open(image_file)
    image_cv = np.array(image_pil.convert('RGB'))
    image = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    custom_config = r'--oem 3 --psm 6'    
    text = pytesseract.image_to_string(thresh, lang="eng", config=custom_config)
        
    return text
    
if __name__ == '__main__':
    print("Modul OCR siap digunakan")
