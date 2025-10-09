"""Streamlit web interface for Materials Property Predictor."""
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.processors.composition_parser import get_example_formulas, CompositionParser

# Page configuration
st.set_page_config(
    page_title="Materials Property Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")


def main():
    """Main application."""
    st.title("🔬 Materials Property Predictor")
    st.markdown("""
    Predict material properties from chemical composition using Artificial Neural Networks.
    Enter a chemical formula to get predictions with uncertainty estimates and feature importance explanations.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        property_name = st.selectbox(
            "Property to Predict",
            ["band_gap", "formation_energy", "density"],
            help="Select the material property to predict"
        )
        
        include_uncertainty = st.checkbox(
            "Include Uncertainty",
            value=True,
            help="Show prediction confidence intervals"
        )
        
        include_explanation = st.checkbox(
            "Include Feature Importance",
            value=True,
            help="Show which features influence the prediction"
        )
        
        st.markdown("---")
        st.markdown("### 📚 Example Formulas")
        examples = get_example_formulas()
        for ex in examples[:5]:
            if st.button(ex, key=f"ex_{ex}"):
                st.session_state.formula_input = ex
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input form
        st.subheader("📝 Enter Chemical Formula")
        
        formula = st.text_input(
            "Chemical Formula",
            value=st.session_state.get("formula_input", ""),
            placeholder="e.g., SiO2, Fe2O3, CaTiO3",
            help="Enter a valid chemical formula using element symbols and subscripts"
        )
        
        # Validation feedback
        if formula:
            is_valid, error_msg = CompositionParser.validate_formula(formula)
            if not is_valid:
                st.error(f"❌ {error_msg}")
            else:
                st.success("✅ Valid formula")
        
        predict_button = st.button("🚀 Predict Properties", type="primary", disabled=not formula)
    
    with col2:
        st.subheader("ℹ️ About")
        st.info("""
        This tool uses machine learning to predict material properties based on chemical composition.
        
        **Supported Properties:**
        - Band Gap (eV)
        - Formation Energy (eV/atom)
        - Density (g/cm³)
        
        **Features:**
        - Neural network predictions
        - Uncertainty quantification
        - Feature importance analysis
        """)
    
    # Make prediction
    if predict_button and formula:
        make_prediction(
            formula,
            property_name,
            include_uncertainty,
            include_explanation
        )


def make_prediction(
    formula: str,
    property_name: str,
    include_uncertainty: bool,
    include_explanation: bool
):
    """Make prediction and display results."""
    with st.spinner("🔄 Making prediction..."):
        try:
            # Call API
            response = requests.post(
                f"{API_URL}/api/v1/predict",
                json={
                    "formula": formula,
                    "properties": [property_name],
                    "include_uncertainty": include_uncertainty,
                    "include_explanation": include_explanation
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                display_results(result, include_uncertainty, include_explanation)
            else:
                error_data = response.json()
                st.error(f"❌ Prediction failed: {error_data.get('detail', 'Unknown error')}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the backend is running.")
            st.info("Start the backend with: `python backend/api/main.py`")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def display_results(
    result: Dict,
    include_uncertainty: bool,
    include_explanation: bool
):
    """Display prediction results."""
    st.success("✅ Prediction completed!")
    
    # Main prediction
    st.subheader("🎯 Prediction Results")
    
    prediction = result.get("prediction", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"{prediction['property_name'].replace('_', ' ').title()}",
            value=f"{prediction['value']:.3f} {prediction['units']}"
        )
    
    with col2:
        if include_uncertainty and prediction.get("uncertainty"):
            st.metric(
                label="Uncertainty (±)",
                value=f"{prediction['uncertainty']:.3f} {prediction['units']}"
            )
    
    with col3:
        st.metric(
            label="Processing Time",
            value=f"{result.get('processing_time', 0):.3f} s"
        )
    
    # Confidence interval
    if include_uncertainty and prediction.get("confidence_interval"):
        st.markdown("**95% Confidence Interval:**")
        ci = prediction["confidence_interval"]
        st.markdown(f"`{ci[0]:.3f}` to `{ci[1]:.3f}` {prediction['units']}")
    
    # Composition
    st.markdown("---")
    st.subheader("🧪 Composition")
    
    composition = result.get("composition", {})
    if composition:
        comp_df = pd.DataFrame([
            {"Element": elem, "Fraction": frac}
            for elem, frac in composition.items()
        ])
        
        fig = px.pie(
            comp_df,
            values="Fraction",
            names="Element",
            title="Elemental Composition"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    if include_explanation and result.get("feature_importance"):
        st.markdown("---")
        st.subheader("📊 Feature Importance")
        st.markdown("*Features that most influence this prediction:*")
        
        display_feature_importance(result["feature_importance"])


def display_feature_importance(feature_importance: List[Dict]):
    """Display feature importance visualization."""
    # Create dataframe
    df = pd.DataFrame(feature_importance)
    
    # Bar chart
    fig = go.Figure()
    
    colors = ['#FF6B6B' if val < 0 else '#4ECDC4' 
              for val in df.get('shap_value', df['importance_score'])]
    
    fig.add_trace(go.Bar(
        y=df['feature_name'],
        x=df.get('shap_value', df['importance_score']),
        orientation='h',
        marker=dict(color=colors),
        text=df.get('shap_value', df['importance_score']).round(3),
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Top Features by Importance",
        xaxis_title="SHAP Value / Importance Score",
        yaxis_title="Feature",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Table with descriptions
    st.markdown("**Feature Details:**")
    display_df = df[['feature_name', 'importance_score']].copy()
    
    if 'description' in df.columns:
        display_df['description'] = df['description']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_batch_prediction():
    """Show batch prediction interface."""
    st.subheader("📦 Batch Prediction")
    
    formulas_text = st.text_area(
        "Enter formulas (one per line)",
        height=150,
        placeholder="SiO2\nFe2O3\nCaTiO3"
    )
    
    if st.button("Predict All"):
        formulas = [f.strip() for f in formulas_text.split('\n') if f.strip()]
        
        if not formulas:
            st.warning("Please enter at least one formula")
            return
        
        with st.spinner(f"Processing {len(formulas)} formulas..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/v1/batch-predict",
                    json={
                        "formulas": formulas,
                        "properties": ["band_gap"],
                        "include_uncertainty": False,
                        "include_explanation": False
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    results = response.json()["results"]
                    display_batch_results(results)
                else:
                    st.error(f"Batch prediction failed: {response.json()}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")


def display_batch_results(results: List[Dict]):
    """Display batch prediction results."""
    # Create summary dataframe
    data = []
    for result in results:
        if result.get("success"):
            pred = result["prediction"]
            data.append({
                "Formula": result["formula"],
                "Property": pred["property_name"],
                "Value": pred["value"],
                "Units": pred["units"]
            })
        else:
            data.append({
                "Formula": result["formula"],
                "Property": "Error",
                "Value": None,
                "Units": result.get("error", "Unknown")
            })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()