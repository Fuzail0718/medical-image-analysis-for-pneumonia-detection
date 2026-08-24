# app_fixed.py
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageEnhance
import json
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set page config FIRST
st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .normal-prediction {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .pneumonia-prediction {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .confidence-bar {
        height: 25px;
        background-color: #e9ecef;
        border-radius: 12px;
        margin: 8px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 25px;
    }
</style>
""", unsafe_allow_html=True)


class PneumoniaDetector:
    def __init__(self):
        self.model = None
        self.class_names = ['NORMAL', 'PNEUMONIA']
        self.model_metrics = None
        self.load_model()
        self.prediction_history = []

    def load_model(self):
        """Load the trained model"""
        try:
            # Try to load the model
            if os.path.exists('pneumonia_model.h5'):
                self.model = keras.models.load_model('pneumonia_model.h5')
                st.sidebar.success("✅ Model loaded successfully!")
            elif os.path.exists('best_pneumonia_model.h5'):
                self.model = keras.models.load_model('best_pneumonia_model.h5')
                st.sidebar.success("✅ Best model loaded successfully!")
            else:
                st.sidebar.error(
                    "❌ No model file found. Please train the model first.")
                return False

            # Try to load metrics
            if os.path.exists('training_results.json'):
                with open('training_results.json', 'r') as f:
                    self.model_metrics = json.load(f)
                st.sidebar.info("📊 Model metrics loaded!")

            return True
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            return False

    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        try:
            # Convert to numpy array
            img_array = np.array(image)

            # Handle different image formats
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array]*3, axis=-1)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = img_array[:, :, :3]

            # Resize and normalize
            img_resized = Image.fromarray(img_array).resize((224, 224))
            img_array = np.array(img_resized) / 255.0

            return np.expand_dims(img_array, axis=0)
        except Exception as e:
            st.error(f"❌ Image processing error: {e}")
            return None

    def predict(self, image, image_name):
        """Make prediction on image"""
        if self.model is None:
            st.error("❌ Model not loaded. Cannot make prediction.")
            return None

        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return None

        try:
            start_time = time.time()
            predictions = self.model.predict(processed_image, verbose=0)
            prediction_time = time.time() - start_time

            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            all_probabilities = {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }

            # Store prediction history
            prediction_record = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'image_name': image_name,
                'predicted_class': self.class_names[predicted_class_idx],
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'prediction_time': prediction_time
            }
            self.prediction_history.append(prediction_record)

            return {
                'predicted_class': self.class_names[predicted_class_idx],
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'prediction_time': prediction_time
            }
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return None


def create_confidence_chart(probabilities, predicted_class):
    """Create a bar chart for confidence scores"""
    fig, ax = plt.subplots(figsize=(8, 4))

    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = ['#28a745' if cls == 'NORMAL' else '#dc3545' for cls in classes]

    # Highlight predicted class
    for i, cls in enumerate(classes):
        if cls == predicted_class:
            colors[i] = '#007bff'

    bars = ax.bar(classes, probs, color=colors, alpha=0.8)
    ax.set_ylabel('Confidence Score')
    ax.set_ylim(0, 1)
    ax.set_title('AI Confidence Scores')

    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    return fig


def main():
    # Initialize detector
    detector = PneumoniaDetector()

    # Header
    st.markdown('<div class="main-header">🏥 AI-Powered Pneumonia Detection</div>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🔧 Control Panel")

        # Model status
        if detector.model is not None:
            st.success("✅ **Model Status:** Active")
            st.info(f"🎯 **Classes:** {', '.join(detector.class_names)}")

            # Show model metrics if available
            if detector.model_metrics:
                st.subheader("📊 Model Performance")
                val_metrics = detector.model_metrics.get(
                    'validation_metrics', {})
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Accuracy", f"{val_metrics.get('accuracy', 0)*100:.1f}%")
                with col2:
                    st.metric(
                        "F1-Score", f"{val_metrics.get('f1_score', 0)*100:.1f}%")
        else:
            st.error("❌ **Model Status:** Not Available")
            st.info("Please run the training script first to generate the model.")

        # Features
        st.header("🛠️ Features")
        show_chart = st.checkbox("Show Confidence Chart", value=True)
        show_history = st.checkbox("Show Prediction History", value=True)

        # Image enhancement
        st.header("🎨 Image Tools")
        enhance_image = st.checkbox("Enhance Image Contrast")

        # Quick actions
        st.header("⚡ Quick Actions")
        if st.button("Clear History"):
            detector.prediction_history = []
            st.success("History cleared!")
            st.rerun()

    # Main content - TWO COLUMNS
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload & Analyze")

        # File upload
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )

        if uploaded_file is not None:
            # Load and display image
            try:
                image = Image.open(uploaded_file)

                # Image enhancement
                if enhance_image:
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1.5)
                    st.image(image, caption="Enhanced Image",
                             use_column_width=True)
                else:
                    st.image(image, caption="Original Image",
                             use_column_width=True)

                # Analysis button
                if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing the image..."):
                        # Simple progress bar
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)

                        # Get prediction
                        prediction_result = detector.predict(
                            image, uploaded_file.name)

                        if prediction_result:
                            # Display results in the same column
                            display_results(prediction_result, show_chart)

                        st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"❌ Error loading image: {e}")

    with col2:
        st.header("📊 Results & Analytics")

        # Show welcome message if no predictions yet
        if not detector.prediction_history:
            display_welcome_message(detector)
        else:
            # Show latest prediction summary
            latest_pred = detector.prediction_history[-1]
            st.subheader("Latest Prediction")
            st.write(f"**Image:** {latest_pred['image_name']}")
            st.write(f"**Result:** {latest_pred['predicted_class']}")
            st.write(f"**Confidence:** {latest_pred['confidence']:.1%}")
            st.write(f"**Time:** {latest_pred['timestamp']}")

            # Show history if enabled
            if show_history:
                display_history(detector.prediction_history)


def display_results(prediction_result, show_chart):
    """Display prediction results"""
    predicted_class = prediction_result['predicted_class']
    confidence = prediction_result['confidence']
    all_probabilities = prediction_result['all_probabilities']
    prediction_time = prediction_result['prediction_time']

    # Prediction box
    box_class = "normal-prediction" if predicted_class == "NORMAL" else "pneumonia-prediction"
    emoji = "✅" if predicted_class == "NORMAL" else "🚨"

    st.markdown(f"""
    <div class="prediction-box {box_class}">
        <h2>{emoji} Prediction: {predicted_class}</h2>
        <h3>📊 Confidence: {confidence:.2%}</h3>
        <p>⏱️ Processing Time: {prediction_time:.3f}s</p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bars
    st.subheader("📈 Confidence Breakdown")
    for class_name, prob in all_probabilities.items():
        col1, col2, col3 = st.columns([2, 5, 1])
        with col1:
            st.write(f"{class_name}:")
        with col2:
            color = "#28a745" if class_name == "NORMAL" else "#dc3545"
            if class_name == predicted_class:
                color = "#007bff"

            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {prob*100}%; background-color: {color};">
                    {prob:.2%}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.write(f"{prob:.2%}")

    # Confidence chart
    if show_chart:
        st.subheader("📊 Confidence Visualization")
        chart_fig = create_confidence_chart(all_probabilities, predicted_class)
        st.pyplot(chart_fig)

    # Medical interpretation
    st.subheader("💡 Clinical Interpretation")
    if predicted_class == "NORMAL":
        st.success("""
        **Normal Chest X-Ray Findings:**
        - No signs of pneumonia detected
        - Clear lung fields observed
        - Normal cardiomediastinal contour
        
        **Recommendations:**
        - Continue regular health monitoring
        - No immediate intervention required
        """)
    else:
        st.error("""
        **Pneumonia Signs Detected:**
        - Areas of consolidation observed
        - Possible air bronchograms
        - Lung opacity patterns consistent with pneumonia
        
        **Urgent Recommendations:**
        - Consult with a healthcare professional immediately
        - Consider follow-up imaging
        - Clinical correlation required for diagnosis
        """)


def display_history(history):
    """Display prediction history"""
    st.subheader("📋 Prediction History")

    # Show last 5 predictions
    for record in history[-5:]:
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.write(f"**{record['image_name']}**")
            with col2:
                emoji = "✅" if record['predicted_class'] == 'NORMAL' else "🚨"
                st.write(f"{emoji} {record['predicted_class']}")
            with col3:
                st.write(f"{record['confidence']:.1%}")
            st.caption(f"Time: {record['timestamp']}")
            st.markdown("---")


def display_welcome_message(detector):
    """Display welcome message when no predictions exist"""
    st.info("""
    ## 🎯 Welcome to AI Pneumonia Detection!
    
    **To get started:**
    
    1. **Upload a chest X-ray image** using the file uploader on the left
    2. **Click 'Analyze Image'** to run the AI analysis
    3. **View results** including:
       - Pneumonia detection prediction
       - Confidence scores
       - Clinical interpretation
       - Medical recommendations
    
    **Supported formats:** PNG, JPG, JPEG
    
    **Note:** This tool is for educational purposes and should not replace professional medical diagnosis.
    """)

    # Show file requirements
    st.warning("""
    **Required Files Check:**
    - ✅ Model file: `pneumonia_model.h5` or `best_pneumonia_model.h5`
    - ✅ Training results: `training_results.json` (optional)
    """)

    # Check files
    col1, col2, col3 = st.columns(3)
    with col1:
        if os.path.exists('pneumonia_model.h5') or os.path.exists('best_pneumonia_model.h5'):
            st.success("✅ Model file found")
        else:
            st.error("❌ Model file missing")

    with col2:
        if os.path.exists('training_results.json'):
            st.success("✅ Training results found")
        else:
            st.warning("⚠️ Training results missing")

    with col3:
        if detector.model is not None:
            st.success("✅ Model loaded")
        else:
            st.error("❌ Model not loaded")


if __name__ == "__main__":
    main()
