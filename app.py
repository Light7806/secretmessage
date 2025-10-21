import streamlit as st
import numpy as np
from PIL import Image
import io  # Needed for file handling in memory

# --- This is our ENCODER logic ---
# (Slightly modified to work with Streamlit's file uploader)

def message_to_binary(message):
    """Converts a string message into a binary string with a stop delimiter."""
    message += "||END||"  # Our special stop delimiter
    binary_message = ''.join([format(ord(char), '08b') for char in message])
    return binary_message

def encode_image(image, secret_message):
    """Hides a secret message within an image using LSB steganography."""
    try:
        # 1. Convert image to a NumPy matrix
        image_matrix = np.array(image.convert('RGB'))
        
        height, width, channels = image_matrix.shape
        total_pixels = height * width
        
        # 2. Convert secret message to a binary vector
        binary_message = message_to_binary(secret_message)
        message_length = len(binary_message)
        
        # 3. Check if the message will fit
        max_bits = total_pixels * channels
        if message_length > max_bits:
            st.error(f"Error: Message is too long for this image. "
                     f"Max bits: {max_bits}, Message bits: {message_length}")
            return None

        # --- 4. The Matrix Cypher (Encoding) ---
        message_bits = np.array([int(bit) for bit in binary_message])
        
        encoded_matrix = image_matrix.copy()
        flat_matrix = encoded_matrix.ravel() # Unroll the 3D matrix to 1D

        # Step A: Clear the LSB (last bit) of all pixels needed
        # (pixel & 254) is (pixel & 11111110)
        flat_matrix[:message_length] = flat_matrix[:message_length] & 254
        
        # Step B: Write our message bits into the cleared LSB
        # (pixel | message_bit)
        flat_matrix[:message_length] = flat_matrix[:message_length] | message_bits

        # 5. Reshape back to an image
        final_image_matrix = flat_matrix.reshape((height, width, channels))
        secret_image = Image.fromarray(final_image_matrix.astype('uint8'), 'RGB')
        
        return secret_image

    except Exception as e:
        st.error(f"An error occurred during encoding: {e}")
        return None

# --- This is our DECODER logic ---

def decode_image(image):
    """Extracts a hidden message from an image."""
    try:
        # 1. Convert image to matrix and flatten
        image_matrix = np.array(image.convert('RGB'))
        flat_matrix = image_matrix.ravel()

        # --- 2. The Matrix "Key" (Decoding) ---
        # (pixel & 1) extracts the LSB from every single pixel
        secret_vector = flat_matrix & 1
        
        # 3. Rebuild the Message
        binary_message_str = "".join(str(bit) for bit in secret_vector)
        
        message = ""
        stop_delimiter = "||END||"
        
        # Read 8 bits at a time
        for i in range(0, len(binary_message_str), 8):
            byte = binary_message_str[i:i+8]
            
            if len(byte) < 8:
                break
                
            char_code = int(byte, 2)
            message += chr(char_code)
            
            # 4. Check for our stop marker
            if message.endswith(stop_delimiter):
                break
        
        if stop_delimiter in message:
            return message[:-len(stop_delimiter)]  # Return the clean message
        else:
            return None  # No message found
            
    except Exception as e:
        st.error(f"An error occurred during decoding: {e}")
        return None

# --- This is the WEB APP interface (built with Streamlit) ---

st.set_page_config(page_title="Matrix Messenger", layout="wide")
st.title("Project Cypher: The Matrix Messenger 🤫")
st.write("Hide a secret message inside an image file. All powered by NumPy matrix operations.")
st.divider()

# Create two columns for Encode/Decode
col1, col2 = st.columns(2)

# --- ENCODER COLUMN ---
with col1:
    st.header("Encode Message")
    
    # 1. Image Uploader
    carrier_image_file = st.file_uploader("1. Upload your 'carrier' image (PNG, JPG)", type=["png", "jpg", "jpeg"])
    
    # 2. Text Input
    secret_message = st.text_area("2. Enter your secret message", height=150)
    
    # 3. Encode Button
    if st.button("Encode Image", type="primary"):
        if carrier_image_file is not None and secret_message:
            with st.spinner('Hiding message in the matrix...'):
                image = Image.open(carrier_image_file)
                encoded_image = encode_image(image, secret_message)
                
                if encoded_image:
                    st.success("Success! Your message is hidden.")
                    st.image(encoded_image, caption="Your encoded image (looks identical)")
                    
                    # Create an in-memory file for download
                    buf = io.BytesIO()
                    encoded_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    # 4. Download Button
                    st.download_button(
                        label="Download Secret Image",
                        data=byte_im,
                        file_name="secret.png",
                        mime="image/png"
                    )
        else:
            st.warning("Please upload an image AND enter a message first.")

# --- DECODER COLUMN ---
with col2:
    st.header("Decode Message")
    
    # 1. Image Uploader
    secret_image_file = st.file_uploader("1. Upload your 'secret' image (PNG only)", type=["png"])
    
    # 2. Decode Button
    if st.button("Decode Image"):
        if secret_image_file is not None:
            with st.spinner('Searching for secret vector...'):
                image = Image.open(secret_image_file)
                decoded_message = decode_image(image)
                
                if decoded_message:
                    st.success("--- SECRET MESSAGE FOUND ---")
                    st.code(decoded_message)  # Display in a code box
                else:
                    st.error("No message found in this image.")
        else:
            st.warning("Please upload an image first.")
