"""
Streamlit frontend for Brain Tumor Segmentation and Survival Prediction.
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import yaml
import sys
import tempfile
import os

sys.path.append(str(Path(__file__).parent.parent))

from inference.predict import BrainTumorPredictor
from utils.postprocess import create_overlay
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(
    page_title="Brain Tumor Segmentation & Survival Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load predictor model (cached)."""
    try:
        return BrainTumorPredictor(config_path="config.yaml")
    except Exception as e:
        st.error(f"Failed to load predictor: {e}")
        st.info("Please ensure models are trained and available. See README.md for instructions.")
        return None


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumor Segmentation & Survival Prediction</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        st.markdown("""
        ### Steps:
        1. Upload MRI scan (NIfTI format)
        2. Select slice (optional)
        3. View segmentation results
        4. Check survival prediction
        """)
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This application uses **Attention U-Net** for brain tumor segmentation
        and **XGBoost** for survival risk prediction.
        
        **Paper:** Attention U-Net: Learning Where to Look for the Pancreas
        (Oktay et al., 2018)
        """)
    
    # Load predictor
    predictor = load_predictor()
    if predictor is None:
        st.stop()
    
    # File upload
    st.header("📤 Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose a NIfTI file (.nii or .nii.gz)",
        type=['nii', 'gz'],
        help="Upload a brain MRI scan in NIfTI format"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Slice selection
            col1, col2 = st.columns(2)
            with col1:
                slice_idx = st.number_input(
                    "Slice Index (leave empty for middle slice)",
                    min_value=0,
                    value=None,
                    step=1,
                    help="Select which slice to analyze"
                )
            
            with col2:
                if st.button("🔍 Analyze", type="primary"):
                    with st.spinner("Processing MRI scan..."):
                        # Predict
                        results = predictor.predict(
                            tmp_path,
                            slice_idx if slice_idx is not None else None,
                            save_outputs=True
                        )
                        
                        # Store results in session state
                        st.session_state['results'] = results
                        st.session_state['tmp_path'] = tmp_path
                        st.success("Analysis complete!")
            
            # Display results if available
            if 'results' in st.session_state:
                display_results(st.session_state['results'], predictor)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)
        
        finally:
            # Clean up temporary file after a delay
            pass  # Keep file for display


def display_results(results: dict, predictor: BrainTumorPredictor):
    """Display prediction results."""
    st.markdown("---")
    st.header("📊 Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tumor Detected", "✅ Yes" if results['tumor_detected'] else "❌ No")
    
    with col2:
        st.metric("Tumor Volume", f"{results['tumor_volume']:.0f} pixels²")
    
    with col3:
        bbox = results['bounding_box']
        st.metric("Bounding Box", f"{bbox['max_x']-bbox['min_x']} × {bbox['max_y']-bbox['min_y']}")
    
    with col4:
        st.metric("Slice Index", results['slice_index'])
    
    # Images
    st.markdown("---")
    st.subheader("🖼️ Visualization")
    
    if 'tmp_path' in st.session_state:
        try:
            # Load original image
            image_slice, _ = predictor.load_mri_slice(
                Path(st.session_state['tmp_path']),
                results['slice_index']
            )
            
            # Get mask
            prob_mask, binary_mask = predictor.predict_segmentation(image_slice)
            
            # Create overlay
            vis_image = image_slice[-1] if image_slice.shape[0] > 0 else image_slice[0]
            overlay = create_overlay(vis_image, binary_mask)
            
            # Display images
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Original MRI")
                # Normalize for display
                display_img = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-6)
                st.image(display_img, use_container_width=True, clamp=True)
            
            with col2:
                st.subheader("Segmentation Mask")
                st.image(binary_mask, use_container_width=True, clamp=True)
            
            with col3:
                st.subheader("Overlay")
                st.image(overlay, use_container_width=True, clamp=True)
        
        except Exception as e:
            st.error(f"Error displaying images: {e}")
    
    # Survival prediction
    st.markdown("---")
    st.subheader("💊 Survival Risk Prediction")
    
    if results.get('survival_prediction'):
        surv_pred = results['survival_prediction']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            risk_category = surv_pred['risk_category']
            risk_color = {
                'Low Risk': '🟢',
                'Medium Risk': '🟡',
                'High Risk': '🔴'
            }
            st.markdown(f"### {risk_color.get(risk_category, '⚪')} {risk_category}")
        
        with col2:
            probs = surv_pred['probabilities']
            
            # Progress bars
            st.progress(probs['low_risk'], text=f"Low Risk: {probs['low_risk']*100:.1f}%")
            st.progress(probs['medium_risk'], text=f"Medium Risk: {probs['medium_risk']*100:.1f}%")
            st.progress(probs['high_risk'], text=f"High Risk: {probs['high_risk']*100:.1f}%")
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['Low Risk', 'Medium Risk', 'High Risk']
            probabilities = [probs['low_risk'], probs['medium_risk'], probs['high_risk']]
            colors = ['green', 'yellow', 'red']
            
            ax.bar(categories, probabilities, color=colors, alpha=0.7)
            ax.set_ylabel('Probability')
            ax.set_ylim([0, 1])
            ax.set_title('Survival Risk Probabilities')
            
            for i, (cat, prob) in enumerate(zip(categories, probabilities)):
                ax.text(i, prob + 0.02, f'{prob*100:.1f}%', ha='center', va='bottom')
            
            st.pyplot(fig)
    else:
        st.info("Survival prediction model not available.")
    
    # Bounding box details
    st.markdown("---")
    with st.expander("📐 Detailed Bounding Box Information"):
        bbox = results['bounding_box']
        st.json(bbox)
    
    # Download results
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    if 'mask_path' in results and results['mask_path']:
        mask_path = Path(results['mask_path'].replace('/files/', 'outputs/predictions/'))
        if mask_path.exists():
            with open(mask_path, 'rb') as f:
                st.download_button(
                    "Download Mask",
                    f.read(),
                    file_name=f"mask_slice{results['slice_index']}.png",
                    mime="image/png"
                )
    
    if 'overlay_path' in results and results['overlay_path']:
        overlay_path = Path(results['overlay_path'].replace('/files/', 'outputs/predictions/'))
        if overlay_path.exists():
            with open(overlay_path, 'rb') as f:
                st.download_button(
                    "Download Overlay",
                    f.read(),
                    file_name=f"overlay_slice{results['slice_index']}.png",
                    mime="image/png"
                )


if __name__ == "__main__":
    main()






