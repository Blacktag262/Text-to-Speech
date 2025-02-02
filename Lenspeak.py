import cv2
import pytesseract
from gtts import gTTS
import os
import time
import re
from collections import Counter

# Set language for TTS
language = 'en'

# Open the webcam
webcam = cv2.VideoCapture(0)
print("Press 'z' to capture an image.")

def clean_text(text):
    # Basic text cleaning
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    return text.strip()

def extract_keywords(text):
    # Tokenize text and remove common English stopwords
    stopwords = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
        "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
        "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
        "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "having", "do", "does", "did", "doing",
        "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "s", "t", "can", "will", "just", "don", "should", "now"
    ])

    words = text.split()
    filtered_words = [word.lower() for word in words if word.lower() not in stopwords]
    keywords_freq = Counter(filtered_words)
    return keywords_freq.most_common(5)  # Get top 5 keywords

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Find contours in the original image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding box for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img = img[y:y+h, x:x+w]  # Crop to the bounding box

    # Convert the cropped area to grayscale
    gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresh_cropped = cv2.threshold(gray_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply binary inversion
    inverted_thresh = cv2.bitwise_not(thresh_cropped)

    # Save and return the preprocessed image
    preprocessed_image_path = 'preprocessed_image.jpg'
    cv2.imwrite(preprocessed_image_path, inverted_thresh)
    return preprocessed_image_path

try:
    while True:
        # Capture frame from webcam
        check, frame = webcam.read()
        if not check:
            print("Failed to capture image.")
            break

        cv2.imshow("Capturing", frame)

        # Wait for user input
        key = cv2.waitKey(1)
        if key == ord('z'):
            # Save the captured image
            img_path = 'saved_img.jpg'
            cv2.imwrite(img_path, frame)
            print("Image saved!")

            # Preprocess the image for better OCR accuracy
            preprocessed_img_path = preprocess_image(img_path)

            # Perform OCR on the preprocessed image
            extracted_text = pytesseract.image_to_string(preprocessed_img_path)
            print("Extracted text:", extracted_text)

            # Clean the extracted text
            cleaned_text = clean_text(extracted_text)
            print("Cleaned text:", cleaned_text)

            # Extract keywords
            keywords = extract_keywords(cleaned_text)
            print("Keywords:", keywords)

            # Text-to-speech with gTTS
            tts = gTTS(text=cleaned_text, lang=language, slow=False)
            tts.save("extracted_text.mp3")

            # Play the audio
            #os.system("start extracted_text.mp3")  # For Windows
            # For Linux you might use: os.system
            os.system("cvlc extracted_text.mp3 &")
            # For macOS you might use: os.system("afplay extracted_text.mp3 &")

            # Optional delay to ensure audio plays
            time.sleep(1)
            break

except KeyboardInterrupt:
    print("Turning off camera.")

finally:
    # Ensure proper release of resources
    webcam.release()
    cv2.destroyAllWindows()
    print("Camera off.")

print("Program ended.")