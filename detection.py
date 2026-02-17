import cv2
import easyocr
import re
from ultralytics import YOLO
import numpy as np

class GateDetector:
    def __init__(self, model_path='license_plate_detector.pt'):
        """
        Initialize YOLO model and EasyOCR reader.
        """
        self.model = YOLO(model_path)
        # Initialize EasyOCR for English. 
        self.reader = easyocr.Reader(['en'], gpu=False) 
        
        # Regex to capture potential plates
        self.BROAD_PLATE_REGEX = r'[A-Z]{2}[0-9]{1,2}[A-Z0-9]{0,3}[0-9]{3,4}'

    def order_points(self, pts):
        """
        Order points: top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):
        """
        Perspective transform to flatten the plate.
        """
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def straighten_plate(self, plate_img):
        """
        Try to find the plate contour and warp it.
        """
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 75, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        display_cnt = None
        
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                display_cnt = approx
                break
        
        if display_cnt is not None:
            warped = self.four_point_transform(plate_img, display_cnt.reshape(4, 2))
            return warped
        
        return plate_img

    def preprocess_image(self, img):
        """
        Enhance image for OCR.
        """
        # 1. Perspective Transform (Attempt to flatten)
        # Only apply if image is large enough, otherwise strict crop is safer
        if img.shape[1] > 60 and img.shape[0] > 15:
             try:
                img = self.straighten_plate(img)
             except:
                 pass # Fallback to original

        # 2. Upscale (Super-resolution heuristic) - vital for long range
        # Resize to a target height of ~100px for OCR
        scale = 3
        if img.shape[0] < 100:
             scale = 100 / img.shape[0]
        
        if scale > 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 3. Contrast Enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 4. Remove Noise
        enhanced = cv2.bilateralFilter(enhanced, 11, 17, 17)
        
        return enhanced

    def correct_characters(self, text):
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        if 8 <= len(clean) <= 10:
            chars = list(clean)
            dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'Z': '2', 'B': '8'}
            dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '2': 'Z', '8': 'B'}
            
            for i in range(2):
                if chars[i] in dict_int_to_char:
                    chars[i] = dict_int_to_char[chars[i]]
            for i in range(2, 4):
                 if chars[i] in dict_char_to_int:
                    chars[i] = dict_char_to_int[chars[i]]
            for i in range(len(chars)-4, len(chars)):
                if chars[i] in dict_char_to_int:
                    chars[i] = dict_char_to_int[chars[i]]
            for i in range(4, len(chars)-4):
                if chars[i] in dict_int_to_char:
                    chars[i] = dict_int_to_char[chars[i]]
            return "".join(chars)
        return clean

    def read_plate_text(self, plate_img):
        """
        Perform OCR on the plate image.
        """
        processed_img = self.preprocess_image(plate_img)
        
        results = self.reader.readtext(
            processed_img, 
            detail=1,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        texts = []
        for (_, text, prob) in results:
            if prob > 0.3:
                texts.append(text)

        if not texts:
            return None

        combined = ''.join(texts)
        corrected = self.correct_characters(combined)
        match = re.search(self.BROAD_PLATE_REGEX, corrected)
        if match:
            return match.group()

        return None

    def detect_and_recognize(self, frame):
        """
        Detects plates and performs OCR.
        """
        # Run YOLO on the full frame
        results = self.model(frame, conf=0.35, verbose=False) # Slightly lower conf for long range
        detections = []
        
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                   continue

                plate_roi = frame[y1:y2, x1:x2]
                if plate_roi.size == 0:
                    continue

                plate_text = self.read_plate_text(plate_roi)
                
                if plate_text:
                    detections.append({
                        'plate': plate_text,
                        'conf': conf,
                        'bbox': (x1, y1, x2, y2),
                        'vehicle_type': 'plate' 
                    })
        
        return detections
