"""
Streamlit UI for Materials Property Predictor
Connects to FastAPI backend for predictions
"""
import streamlit as st
import requests
import json
import pandas as pd

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Materials Property Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔬 Materials Property Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered prediction of material properties using trained neural networks</div>', unsafe_allow_html=True)

# Check API health
def check_api():
    try:
        response = requests.get(f"{API_URL}/api/v1/health", timeout=3)
        return response.json()
    except:
        return None

api_status = check_api()

if api_status is None:
    st.error("❌ **API Server Not Running**")
    st.info("""
    **How to start the server:**
    
    1. Open a terminal in your project root directory
    2. Run: `python start.py`
    
    Or manually:
    - Terminal 1: `python api.py`
    - Terminal 2: `streamlit run frontend/app.py`
    """)
    st.stop()

available_properties = api_status.get("models_loaded", [])

if not available_properties:
    st.warning("⚠️ **API is running but no models are loaded**")
    st.info("Please ensure your `models/` folder contains trained model files (`.pt`) and their corresponding scalers (`.pkl`)")
    st.stop()

# Success message
st.success(f"✅ **Connected to API** | {len(available_properties)} model(s) loaded")

# Main input section
st.markdown("---")
st.subheader("🧪 Make a Prediction")

col1, col2 = st.columns([2, 1])

if 'formula' not in st.session_state:
    st.session_state.formula = ""

with col1:
    formula = st.text_input(
        "Chemical Formula", 
        key ="formula",
        placeholder="e.g., Si, TiO2, CuFeO2, La0.5Sr0.5MnO3",
        help="Enter a valid chemical formula. Examples: Si, TiO2, BaTiO3, La0.5Sr0.5MnO3"
    )

with col2:
    # Create a nice display name for properties
    property_display = {prop: prop.replace("_", " ").title() for prop in available_properties}
    
    selected_display = st.selectbox(
        "Property to Predict",
        options=list(property_display.values()),
        help="Select the material property you want to predict"
    )
    
    # Get the actual property name
    property_name = [k for k, v in property_display.items() if v == selected_display][0]

# Predict button
predict_col1, predict_col2, predict_col3 = st.columns([2, 1, 2])
with predict_col2:
    predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)

if predict_button:
    if not formula.strip():
        st.error("⚠️ Please enter a chemical formula")
    else:
        with st.spinner("🔄 Making prediction..."):
            try:
                # Make prediction request
                response = requests.post(
                    f"{API_URL}/api/v1/predict",
                    json={"formula": formula.strip(), "property": property_name},
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.markdown("---")
                    st.success("✅ **Prediction Successful!**")
                    
                    # Main metrics
                    st.subheader("📊 Results")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(
                            label="Predicted Value",
                            value=f"{result['value']:.6f}",
                            help="The predicted value for the selected property"
                        )
                    
                    with metric_col2:
                        st.metric(
                            label="Uncertainty (±)",
                            value=f"{result['uncertainty']:.6f}",
                            help="Model uncertainty (standard deviation)"
                        )
                    
                    with metric_col3:
                        ci = result['confidence_interval']
                        st.metric(
                            label="95% Confidence Interval",
                            value=f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                            help="95% confidence interval for the prediction"
                        )
                    
                    # Composition breakdown
                    st.subheader("🧬 Composition Analysis")
                    comp_data = result.get('composition', {})
                    
                    if comp_data:
                        # Create DataFrame for better visualization
                        comp_df = pd.DataFrame([
                            {"Element": elem, "Fraction": f"{frac:.4f}", "Percentage": f"{frac*100:.2f}%"}
                            for elem, frac in comp_data.items()
                        ])
                        
                        col_left, col_right = st.columns([1, 1])
                        
                        with col_left:
                            st.dataframe(comp_df, use_container_width=True, hide_index=True)
                        
                        with col_right:
                            # Bar chart
                            st.bar_chart(pd.DataFrame({
                                "Element": list(comp_data.keys()),
                                "Fraction": list(comp_data.values())
                            }).set_index("Element"))
                    
                    # Additional details
                    with st.expander("🔍 View Detailed Response"):
                        st.json(result)
                    
                    # Download results
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json.dumps(result, indent=2),
                        file_name=f"prediction_{formula}_{property_name}.json",
                        mime="application/json"
                    )
                    
                else:
                    error_detail = response.json().get('detail', 'Unknown error occurred')
                    st.error(f"❌ **Prediction Failed:** {error_detail}")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ **Request Timed Out** - The server took too long to respond. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 **Connection Error** - Could not connect to the API server.")
            except Exception as e:
                st.error(f"❌ **Error:** {str(e)}")

# Sidebar
with st.sidebar:
    st.header("📚 Available Models")
    st.markdown("The following properties can be predicted:")
    
    for prop in sorted(available_properties):
        display_name = prop.replace("_", " ").title()
        st.markdown(f"- **{display_name}** (`{prop}`)")
    
    st.divider()
    
    st.header("💡 Example Formulas")
    st.markdown("Click to use these examples:")
    
    examples = {
        "Silicon": "Si",
        "Titanium Dioxide": "TiO2",
        "Copper Iron Oxide": "CuFeO2",
        "Lanthanum Strontium Manganite": "La0.5Sr0.5MnO3",
        "Barium Titanate": "BaTiO3",
        "Iron Oxide": "Fe2O3",
        "Aluminum Oxide": "Al2O3",
        "Gallium Nitride": "GaN"
    }
    
    for name, formula in examples.items():
        if st.button(f"{name} ({formula})", use_container_width=True, key=f"ex_{formula}"):
            st.session_state.formula = formula
            st.rerun()
    
    st.divider()
    
    st.header("ℹ️ About")
    st.markdown("""
    This application uses trained artificial neural networks (ANNs) to predict 
    material properties from chemical formulas.
    
    **Features:**
    - Multiple property predictions
    - Uncertainty quantification
    - Composition analysis
    - Easy-to-use interface
    
    """)
    
    st.divider()
    
    # API status
    st.header("🔧 System Status")
    if api_status:
        st.success("✅ API: Running")
        st.info(f"📊 Models: {len(available_properties)}")
    
    with st.expander("View API Info"):
        st.json(api_status)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666;">Made with ❤️ using Streamlit and FastAPI</div>',
    unsafe_allow_html=True
)