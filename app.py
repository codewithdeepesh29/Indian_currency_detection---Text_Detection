import cv2
import pytesseract
import requests
import streamlit as st
from PIL import Image
import pyttsx3  # Import pyttsx3 module

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Function to call the API for Indian currency denomination recognition
def recognize_currency(image):
    # Your API endpoint for currency recognition
    api_endpoint = 'http://127.0.0.1:5000/classify_denomination'

    # Encode image to base64
    _, img_encoded = cv2.imencode('.jpg', image)
    img_base64 = img_encoded.tobytes()

    # Send POST request to the API
    response = requests.post(api_endpoint, data=img_base64)

    # Check if request was successful
    if response.status_code == 200:
        return response.text
    else:
        st.error("Error: Unable to recognize currency.")
        return ""

# Function to convert text to speech
def convert_text_to_speech(text):
    # Convert the text to speech
    engine.say(text)
    engine.runAndWait()

def main():
    st.title("Currency Recognition")

    st.sidebar.title("Options")
    option = st.sidebar.radio("Choose an option:", ("Camera", "Image Upload"))

    if option == "Camera":
        st.write("### Live Camera")
        run_camera()
    elif option == "Image Upload":
        st.write("### Upload Image")
        run_image_upload()

def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Unable to access camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Unable to capture frame.")
            break

        st.image(frame, channels="BGR")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            text = pytesseract.image_to_string(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), lang='eng')
            st.write("Detected Text:")
            st.write(text)
            convert_text_to_speech(text)
        elif key == ord('a'):
            currency_text = recognize_currency(frame)
            st.write("Currency Recognition:")
            st.write(currency_text)
            convert_text_to_speech(currency_text)

    cap.release()
    cv2.destroyAllWindows()

def run_image_upload():
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        text = pytesseract.image_to_string(cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY), lang='eng')
        st.write("Detected Text:")
        st.write(text)
        convert_text_to_speech(text)

        currency_text = recognize_currency(image_array)
        st.write("Currency Recognition:")
        st.write(currency_text)
        convert_text_to_speech(currency_text)

if __name__ == "__main__":
    main()