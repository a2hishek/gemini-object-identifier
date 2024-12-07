import streamlit as st
import google.generativeai as genai
import os

# Configure the Gemini API
genai.configure(api_key="Your_Gemini_API_Key")

# Function to upload the file to Gemini
def upload_to_gemini(path, mime_type=None):
    """
    Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        st.success(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file.uri
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return None

# Function to run the model and process the uploaded image
def analyze_image(file_uri):
    """Analyzes the image using the Gemini model."""
    generation_config = {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Construct the prompt
    prompt = f"Extract the objects in the provided image and output them in a list in alphabetical order.\nImage: {file_uri}\nList of Objects:"
    
    try:
        response = model.generate_content([prompt])
        return response.text
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

# Streamlit UI
st.title("Image Object Detection with Gemini")
st.write("Upload an image to analyze its content and extract a list of objects.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded file temporarily
    temp_file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.image(temp_file_path, caption="Uploaded Image", use_column_width=True)
    
    # Upload the image to Gemini
    st.write("Processing the image...")
    file_uri = upload_to_gemini(temp_file_path, mime_type="image/jpeg")
    
    if file_uri:
        # Analyze the image
        st.write("Analyzing the image...")
        result = analyze_image(file_uri)
        
        if result:
            st.write("### Detected Objects:")
            st.write(result)

    # Cleanup the temporary file
    os.remove(temp_file_path)

