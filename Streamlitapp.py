import cv2
import pytesseract
import requests
import streamlit as st

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
        print("Error: Unable to recognize currency.")
        return ""

def main():
    st.title("Currency Recognition App")

    # Open the default camera (usually the first camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Unable to access camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Unable to capture frame.")
            break

        # Display the captured frame
        st.image(frame, channels="BGR", caption='Live Camera')

        # Check for user input to close the app, detect text, or call the API
        key = st.sidebar.radio("Select an option", ['Exit', 'Detect Text', 'Recognize Currency'])
        
        if key == 'Exit':
            break
        elif key == 'Detect Text':
            # Detect text when 'Detect Text' is selected
            text = pytesseract.image_to_string(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), lang='eng')
            st.write("Detected Text:")
            st.write(text)
        elif key == 'Recognize Currency':
            # Call the API for currency recognition when 'Recognize Currency' is selected
            currency_text = recognize_currency(frame)
            st.write("Currency Recognition:")
            st.write(currency_text)

    # Release the camera
    cap.release()

if __name__ == "__main__":
    main()
