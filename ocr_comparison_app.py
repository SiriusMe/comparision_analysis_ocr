import streamlit as st
# Set page config first
st.set_page_config(page_title="OCR Comparison", layout="wide")

import easyocr
import pytesseract
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import os
import warnings
import fitz  # PyMuPDF for PDF
import tempfile
import pandas as pd
import logging
import platform
import gc
import torch
import time
from contextlib import contextmanager
import tensorflow as tf

# Initialize logger before imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize TensorFlow and Keras OCR with error handling
try:
    import keras_ocr
    KERAS_OCR_AVAILABLE = True
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
except Exception as e:
    KERAS_OCR_AVAILABLE = False
    st.warning("Keras OCR is not available. The app will run with reduced functionality.")
    logger.error(f"Failed to import Keras OCR: {str(e)}")

# Tesseract error handling
try:
    pytesseract.get_tesseract_version()
except Exception as e:
    st.error("Tesseract is not properly installed or configured.")
    logger.error(f"Tesseract configuration error: {str(e)}")

# Modify the Tesseract configuration section
import platform

# Configure Tesseract path based on deployment environment
if os.getenv('STREAMLIT_DEPLOYMENT') or platform.system() != 'Windows':
    # For Streamlit Cloud (Linux environment)
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
else:
    # For local Windows environment
    if os.path.exists(r'Tesseract-OCR\tesseract.exe'):
        pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'
    else:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Add Tesseract version check
try:
    tesseract_version = pytesseract.get_tesseract_version()
    logger.info(f"Tesseract version: {tesseract_version}")
except Exception as e:
    st.error("""
        Tesseract is not properly configured. 
        If you're running locally, please install Tesseract-OCR. 
        If this is on Streamlit Cloud, please contact the administrator.
    """)
    logger.error(f"Tesseract configuration error: {str(e)}")

# Add these imports at the top, after the existing imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# Memory management settings
gc.enable()  # Enable garbage collection
torch.set_grad_enabled(False)  # Disable gradient calculation
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear CUDA cache

# Add this reconnection helper
@contextmanager
def handle_disconnection(max_retries=3, delay=2):
    """Context manager to handle server disconnections"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            yield
            break  # If successful, break out of the loop
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise  # Re-raise the exception if all retries failed
            else:
                logger.warning(f"Attempt {retry_count} failed, retrying in {delay} seconds...")
                time.sleep(delay)
                continue

def find_innermost_boundary(image):
    """Find the innermost boundary rectangle that contains the main drawing"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        height, width = image.shape[:2]
        valid_rectangles = []

        # Process each contour
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            rect_area = w * h

            # Filter valid rectangles
            is_valid = (
                w > width * 0.1 and h > height * 0.1 and
                abs(area - rect_area) / rect_area < 0.4 and
                x >= 0 and y >= 0
            )

            if is_valid:
                valid_rectangles.append({
                    'contour': cnt,
                    'area': area,
                    'rect': (x, y, w, h)
                })

        if not valid_rectangles:
            return None, None, None

        # Sort by area and get the second largest (usually the main drawing area)
        valid_rectangles.sort(key=lambda x: x['area'], reverse=True)
        main_rect = valid_rectangles[1]['rect'] if len(valid_rectangles) > 1 else valid_rectangles[0]['rect']
        main_cnt = valid_rectangles[1]['contour'] if len(valid_rectangles) > 1 else valid_rectangles[0]['contour']

        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [main_cnt], -1, 255, -1)

        # Draw contour on original image
        image_with_contour = image.copy()
        cv2.drawContours(image_with_contour, [main_cnt], -1, (0, 255, 0), 2)

        return mask, main_rect, image_with_contour

    except Exception as e:
        st.error(f"Error in boundary detection: {str(e)}")
        return None, None, None

def convert_pdf_to_image(pdf_file):
    """Convert PDF to image"""
    try:
        logger.info("Starting PDF conversion")
        # Create a temporary file to save the PDF content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
            tmp_pdf.write(pdf_file.getvalue())
            tmp_pdf_path = tmp_pdf.name

        # Open the temporary PDF file
        pdf_document = fitz.open(tmp_pdf_path)
        
        try:
            # Get first page
            page = pdf_document[0]
            
            # Convert to image with higher resolution
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_data = pix.tobytes("png")
            
            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            logger.info("PDF conversion successful")
            return img
        finally:
            # Clean up
            pdf_document.close()
            os.unlink(tmp_pdf_path)
            
    except Exception as e:
        logger.error(f"PDF conversion failed: {str(e)}", exc_info=True)
        st.error(f"Error converting PDF: {str(e)}")
        return None

def convert_dxf_to_image(dxf_file):
    """Convert DXF to image"""
    st.error("DXF support not available. Please install ezdxf package.")
    return None

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all OCR models with caching"""
    models = {
        'easyocr': None,
        'keras': None
    }
    
    # Load EasyOCR with memory-efficient settings
    try:
        with st.spinner('Loading EasyOCR model... This may take a few minutes.'):
            # First check if torch is available
            import torch
            import gc
            
            # Force garbage collection before loading model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Suppress torch warnings
            torch.set_warn_always(False)
            
            # Initialize reader with minimal memory settings
            try:
                models['easyocr'] = easyocr.Reader(
                    ['en'],
                    gpu=False,  # Force CPU mode for stability
                    model_storage_directory=os.path.join(os.getcwd(), 'models'),  # Absolute path
                    download_enabled=True,
                    verbose=False
                )
                logger.info("EasyOCR model loaded successfully")
                
                # Force garbage collection after loading
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as model_error:
                logger.error(f"EasyOCR model initialization failed: {str(model_error)}")
                st.warning("EasyOCR model failed to initialize. Trying minimal configuration...")
                
                # Ultra-minimal fallback configuration
                try:
                    gc.collect()  # Clean memory before retry
                    models['easyocr'] = easyocr.Reader(
                        ['en'],
                        gpu=False,
                        model_storage_directory=os.path.join(os.getcwd(), 'models'),
                        download_enabled=True,
                        verbose=False
                    )
                    logger.info("EasyOCR loaded with minimal configuration")
                except Exception as fallback_error:
                    logger.error(f"EasyOCR fallback initialization failed: {str(fallback_error)}")
                    models['easyocr'] = None
                
    except Exception as e:
        st.warning("EasyOCR failed to load. Some features will be disabled.")
        logger.error(f"EasyOCR loading error: {str(e)}", exc_info=True)
        models['easyocr'] = None
    
    finally:
        # Final garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load Keras OCR if available
    if KERAS_OCR_AVAILABLE:
        try:
            with st.spinner('Loading Keras OCR model...'):
                models['keras'] = keras_ocr.pipeline.Pipeline()
        except Exception as e:
            st.warning("Keras OCR failed to load. Some features will be disabled.")
            logger.error(f"Keras OCR loading error: {str(e)}")
            models['keras'] = None
    
    return models

def preprocess_image(image):
    """Preprocess image for better OCR results"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        return denoised
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def process_with_easyocr(image, reader):
    """Process image with EasyOCR"""
    try:
        if reader is None:
            return None
            
        # Store original image for visualization
        original_image = image.copy()
            
        # Resize image if too large
        max_dimension = 2000  # Maximum dimension threshold
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            original_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Convert to grayscale for OCR processing
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        # Split image into chunks if it's large
        chunk_size = 1000  # Size of each chunk
        height, width = gray_image.shape[:2]
        
        all_results = []
        
        # Process image in chunks if it's large
        if max(height, width) > chunk_size:
            for y in range(0, height, chunk_size):
                for x in range(0, width, chunk_size):
                    # Extract chunk
                    chunk = gray_image[y:min(y+chunk_size, height), x:min(x+chunk_size, width)]
                    
                    # Process chunk with progress indicator
                    with st.spinner(f'Processing region {x}:{x+chunk_size}, {y}:{y+chunk_size}...'):
                        chunk_results = reader.readtext(chunk)
                        
                        # Adjust coordinates to original image space
                        for box, text, conf in chunk_results:
                            adjusted_box = [[p[0] + x, p[1] + y] for p in box]
                            all_results.append([adjusted_box, text, conf])
                    
                    # Force garbage collection after each chunk
                    gc.collect()
        else:
            # Process small image directly
            with st.spinner('Processing image...'):
                all_results = reader.readtext(gray_image)
        
        # Filter for confidence >= 0.6 (60%) and horizontal text
        horizontal_results = []
        for box, text, conf in all_results:
            if conf >= 0.6:  # Check confidence threshold
                # Calculate if text is horizontal based on box dimensions
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                
                # Only include horizontal text (width > height)
                if height <= width * 1.5:
                    horizontal_results.append([box, text, conf])
                    
            # Update progress
            if len(horizontal_results) % 10 == 0:
                st.write(f"Found {len(horizontal_results)} text regions...")
        
        logger.info(f"Total text regions found: {len(horizontal_results)}")
        
        # Return results along with the original color image for visualization
        return horizontal_results, original_image
        
    except Exception as e:
        logger.error(f"Error in EasyOCR processing: {str(e)}", exc_info=True)
        st.error(f"Error in EasyOCR processing: {str(e)}")
        return None, None

def process_with_tesseract(image):
    """Process image with Tesseract"""
    try:
        # Preprocess image
        preprocessed = preprocess_image(image)
        if preprocessed is None:
            return None
        
        # Get results for horizontal text
        data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
        
        # Filter data for confidence >= 70 and horizontal text
        filtered_data = {k: [] for k in data.keys()}
        
        for i in range(len(data['text'])):
            if (int(data['conf'][i]) >= 70 and 
                data['text'][i].strip() and
                data['width'][i] > data['height'][i]):  # Only horizontal text
                
                # Add to filtered data
                for key in data.keys():
                    filtered_data[key].append(data[key][i])
        
        return filtered_data
    except Exception as e:
        st.error(f"Error in Tesseract processing: {str(e)}")
        return None

def process_with_keras_ocr(image, pipeline):
    """Process image with Keras OCR"""
    try:
        if pipeline is None:
            return None
            
        # Convert image to RGB (Keras OCR requirement)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get predictions
        predictions = pipeline.recognize([rgb_image])[0]
        
        # Format results to match EasyOCR format
        results = []
        for text, box in predictions:
            # Calculate if text is horizontal
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            # Only include horizontal text
            if height <= width * 1.5:
                # Use 1.0 as confidence since Keras OCR doesn't provide confidence scores
                results.append([box, text, 1.0])
        
        return results
    except Exception as e:
        st.error(f"Error in Keras OCR processing: {str(e)}")
        return None

def draw_results(image, results, method, boundary_image=None):
    """Draw detection results on image"""
    try:
        if results is None:
            return None
            
        # Unpack results if it's from EasyOCR with modified return
        if method == "EasyOCR" and isinstance(results, tuple):
            results, _ = results  # Ignore the image from results, use the passed image
            
        # Create figure with three subplots: original, boundary, and OCR results
        fig = plt.figure(figsize=(20, 10))
        
        # Create grid for subplots
        if boundary_image is not None:
            gs = fig.add_gridspec(1, 3)
            ax_orig = fig.add_subplot(gs[0, 0])  # Original image
            ax_bound = fig.add_subplot(gs[0, 1])  # Boundary image
            ax_ocr = fig.add_subplot(gs[0, 2])    # OCR results
            
            # Show boundary image
            ax_bound.imshow(cv2.cvtColor(boundary_image, cv2.COLOR_BGR2RGB))
            ax_bound.set_title("Detected Drawing Boundary")
            ax_bound.axis('off')
        else:
            gs = fig.add_gridspec(1, 2)
            ax_orig = fig.add_subplot(gs[0, 0])  # Original image
            ax_ocr = fig.add_subplot(gs[0, 1])    # OCR results
        
        # Show original image
        ax_orig.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_orig.set_title("Original Image")
        ax_orig.axis('off')
        
        # Show OCR results
        ax_ocr.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_ocr.set_title("OCR Results")
        ax_ocr.axis('off')
        
        if method in ["EasyOCR", "Keras OCR"]:
            # Draw horizontal detections in green
            for detection in results:
                box = detection[0]
                text = detection[1]
                conf = detection[2]
                
                # Create polygon for bounding box
                bbox = patches.Polygon(box, linewidth=1, edgecolor='green', facecolor='none')
                ax_ocr.add_patch(bbox)
                
                # Add text with confidence
                ax_ocr.text(box[0][0], box[0][1], 
                        f"{text} ({conf:.2f})" if method == "EasyOCR" else text, 
                        color='green', fontsize=6, bbox=dict(facecolor='white', alpha=0.7))
        
        elif method == "Tesseract":
            # Draw horizontal detections in green
            if len(results['text']) > 0:  # Check if there are any results
                for i in range(len(results['text'])):
                    if results['text'][i].strip():  # Only process non-empty text
                        x = results['left'][i]
                        y = results['top'][i]
                        w = results['width'][i]
                        h = results['height'][i]
                        text = results['text'][i]
                        conf = results['conf'][i]
                        
                        rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                              edgecolor='green', facecolor='none')
                        ax_ocr.add_patch(rect)
                        ax_ocr.text(x, y, f"{text} ({conf})", color='green', fontsize=6,
                                  bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error drawing results: {str(e)}")
        return None

def calculate_statistics(results, method):
    """Calculate detection statistics"""
    try:
        if method == "EasyOCR":
            total_detections = len(results)
            avg_confidence = np.mean([det[2] for det in results]) if results else 0
            
            # Group by confidence ranges
            high_conf = len([det for det in results if det[2] >= 0.8])
            med_conf = len([det for det in results if 0.7 <= det[2] < 0.8])
            low_conf = len([det for det in results if 0.6 <= det[2] < 0.7])
            
            stats = {
                "Total Detections": total_detections,
                "Average Confidence": f"{avg_confidence:.2%}",
                "High Confidence (≥80%)": high_conf,
                "Medium Confidence (70-79%)": med_conf,
                "Low Confidence (60-69%)": low_conf
            }
            
        elif method == "Tesseract":
            total_detections = len(results['text'])
            confidences = [conf for conf in results['conf'] if conf != -1]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Group by confidence ranges
            high_conf = len([conf for conf in confidences if conf >= 80])
            med_conf = len([conf for conf in confidences if 70 <= conf < 80])
            low_conf = len([conf for conf in confidences if 60 <= conf < 70])
            
            stats = {
                "Total Detections": total_detections,
                "Average Confidence": f"{avg_confidence:.2f}%",
                "High Confidence (≥80%)": high_conf,
                "Medium Confidence (70-79%)": med_conf,
                "Low Confidence (60-69%)": low_conf
            }
            
        elif method == "Keras OCR":
            total_detections = len(results)
            avg_confidence = np.mean([conf for conf in results[2]]) if results else 0
            
            stats = {
                "Total Detections": total_detections,
                "Average Confidence": f"{avg_confidence:.2%}"
            }
            
        return stats
    except Exception as e:
        st.error(f"Error calculating statistics: {str(e)}")
        return None

def display_statistics(stats):
    """Display detection statistics in a formatted way"""
    if stats:
        st.subheader("📊 Detection Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🔢 Total Detections", stats["Total Detections"])
            st.metric("Average Confidence", stats["Average Confidence"])
        
        with col2:
            st.metric("High Confidence (≥80%)", stats["High Confidence (≥80%)"])
            st.metric("Medium Confidence (70-79%)", stats["Medium Confidence (70-79%)"])
            st.metric("Low Confidence (60-69%)", stats["Low Confidence (60-69%)"])

def display_detections(results, method):
    """Display all detections in a table format"""
    if not results:
        return
    
    st.subheader("📋 All Detections")
    
    if method == "EasyOCR":
        # Create DataFrame for EasyOCR results
        data = {
            "Text": [det[1] for det in results],
            "Confidence": [f"{det[2]:.2%}" for det in results],
            "Position": [f"({int(det[0][0][0])}, {int(det[0][0][1])})" for det in results]
        }
        df = pd.DataFrame(data)
        df.index = range(1, len(df) + 1)  # 1-based indexing
        
    elif method == "Tesseract":
        # Create DataFrame for Tesseract results
        data = {
            "Text": results['text'],
            "Confidence": [f"{conf}%" for conf in results['conf']],
            "Position": [f"({left}, {top})" for left, top in zip(results['left'], results['top'])]
        }
        df = pd.DataFrame(data)
        df.index = range(1, len(df) + 1)  # 1-based indexing
    
    elif method == "Keras OCR":
        # Create DataFrame for Keras OCR results
        data = {
            "Text": [det[1] for det in results],
            "Position": [f"({int(det[0][0][0])}, {int(det[0][0][1])})" for det in results]
        }
        df = pd.DataFrame(data)
        df.index = range(1, len(df) + 1)  # 1-based indexing
    
    # Display DataFrame with enhanced styling
    st.dataframe(
        df.style.set_properties(**{
            'background-color': '#f0f2f6',
            'color': '#1f1f1f',
            'border-color': '#ffffff',
            'font-size': '16px',
            'padding': '10px',
            'text-align': 'left'
        }).set_table_styles([
            {'selector': 'th',
             'props': [('background-color', '#0e1117'),
                      ('color', '#ffffff'),
                      ('font-weight', 'bold'),
                      ('font-size', '16px'),
                      ('padding', '10px')]},
            {'selector': 'tr:hover',
             'props': [('background-color', '#e6e9ef')]},
        ]),
        use_container_width=True,  # Make table full width
        height=400  # Set fixed height with scrolling
    )

def main():
    st.title("📝 OCR Methods Comparison")
    st.write("Compare text detection results from EasyOCR, Tesseract, and Keras OCR")
    
    try:
        # Initialize session state for models if not exists
        if 'models' not in st.session_state:
            st.session_state.models = None
        
        # Load models with reconnection handling
        with handle_disconnection():
            if st.session_state.models is None:
                with st.spinner('Initializing models...'):
                    st.session_state.models = load_models()
        
        models = st.session_state.models
        
        # Check which models are available
        easyocr_available = models['easyocr'] is not None
        keras_available = models['keras'] is not None
        
        # Show model status
        st.sidebar.subheader("Model Status")
        st.sidebar.write("EasyOCR: " + ("✅ Ready" if easyocr_available else "❌ Not Available"))
        st.sidebar.write("Tesseract: " + ("✅ Ready" if pytesseract.get_tesseract_version() else "❌ Not Available"))
        st.sidebar.write("Keras OCR: " + ("✅ Ready" if keras_available else "❌ Not Available"))
        
        # File uploader with supported file types
        uploaded_file = st.file_uploader(
            "Choose a file...", 
            type=["jpg", "jpeg", "png", "pdf"]
        )
        
        if uploaded_file is not None:
            with handle_disconnection():
                # Process file with automatic reconnection
                try:
                    # Convert file to image based on type
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    
                    if file_type == 'pdf':
                        with st.spinner('Converting PDF...'):
                            image = convert_pdf_to_image(uploaded_file)
                    else:
                        # Read image file directly
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        st.error("Failed to load file. Please try another file.")
                        return
                    
                    # Detect drawing boundary
                    mask, boundary_rect, boundary_image = find_innermost_boundary(image)
                    
                    if mask is not None:
                        # Apply mask to image
                        masked_image = cv2.bitwise_and(image, image, mask=mask)
                    else:
                        masked_image = image
                    
                    # Create tabs and process with available models
                    available_tabs = []
                    if easyocr_available:
                        available_tabs.append("🔍 EasyOCR")
                    available_tabs.append("🎯 Tesseract")
                    if keras_available:
                        available_tabs.append("🤖 Keras OCR")
                    
                    tabs = st.tabs(available_tabs)
                    tab_index = 0
                    
                    # Process with each available model
                    if easyocr_available:
                        with tabs[tab_index]:
                            with handle_disconnection():
                                with st.spinner('Processing with EasyOCR...'):
                                    easyocr_results, processed_image = process_with_easyocr(masked_image, models['easyocr'])
                                    fig = draw_results(processed_image, easyocr_results, "EasyOCR", boundary_image)
                                    if fig:
                                        st.pyplot(fig)
                                    if easyocr_results:
                                        stats = calculate_statistics(easyocr_results, "EasyOCR")
                                        display_statistics(stats)
                                        display_detections(easyocr_results, "EasyOCR")
                        tab_index += 1
                    
                    # Tesseract is always available as a tab
                    with tabs[tab_index]:
                        with handle_disconnection():
                            with st.spinner('Processing with Tesseract...'):
                                tesseract_results = process_with_tesseract(masked_image)
                                fig = draw_results(masked_image, tesseract_results, "Tesseract", boundary_image)
                                if fig:
                                    st.pyplot(fig)
                                if tesseract_results:
                                    stats = calculate_statistics(tesseract_results, "Tesseract")
                                    display_statistics(stats)
                                    display_detections(tesseract_results, "Tesseract")
                    tab_index += 1
                    
                    if keras_available:
                        with tabs[tab_index]:
                            with handle_disconnection():
                                with st.spinner('Processing with Keras OCR...'):
                                    keras_results = process_with_keras_ocr(masked_image, models['keras'])
                                    fig = draw_results(masked_image, keras_results, "Keras OCR", boundary_image)
                                    if fig:
                                        st.pyplot(fig)
                                    if keras_results:
                                        # Since Keras OCR doesn't provide confidence scores, we'll show simpler stats
                                        st.subheader("📊 Detection Statistics")
                                        st.metric("🔢 Total Detections", len(keras_results))
                                        # Display detections
                                        st.subheader("📋 All Detections")
                                        data = {
                                            "Text": [det[1] for det in keras_results],
                                            "Position": [f"({int(det[0][0][0])}, {int(det[0][0][1])})" for det in keras_results]
                                        }
                                        df = pd.DataFrame(data)
                                        df.index = range(1, len(df) + 1)
                                        st.dataframe(
                                            df.style.set_properties(**{
                                                'background-color': '#f0f2f6',
                                                'color': '#1f1f1f',
                                                'border-color': '#ffffff',
                                                'font-size': '16px',
                                                'padding': '10px',
                                                'text-align': 'left'
                                            }).set_table_styles([
                                                {'selector': 'th',
                                                 'props': [('background-color', '#0e1117'),
                                                          ('color', '#ffffff'),
                                                          ('font-weight', 'bold'),
                                                          ('font-size', '16px'),
                                                          ('padding', '10px')]},
                                                {'selector': 'tr:hover',
                                                 'props': [('background-color', '#e6e9ef')]},
                                            ]),
                                            use_container_width=True,
                                            height=400
                                        )
                        
                except Exception as e:
                    logger.error(f"Error processing file: {str(e)}")
                    st.error("An error occurred. Retrying...")
                    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        # Don't show error to user, just retry silently
        time.sleep(2)
        st.experimental_rerun()

if __name__ == "__main__":
    main() 