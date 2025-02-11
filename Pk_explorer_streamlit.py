import numpy as np
import streamlit as st
import plotly.graph_objects as go
from cosmology import plin_new_emulated, sigma8_to_As_max_precision

def k_to_mass(k, Om):
    """Convert wavenumber k to mass scale"""
    return 1.16e12 * Om * k**-3

def main():
    st.set_page_config(layout="wide")
    st.title("Cosmic Power Spectrum Explorer")

    # Sidebar controls
    with st.sidebar:
        st.header("Cosmological Parameters")
        fiducial = {
            "sigma8": st.slider("σ₈", 0.7, 0.9, 0.8102, 0.0001),
            "Om": st.slider("Ω_m", 0.2, 0.7, 0.3111, 0.001),
            "Ob": st.slider("Ω_b", 0.01, 0.1, 0.049, 0.001),
            "h": st.slider("h", 0.5, 0.8, 0.6766, 0.001),
            "ns": st.slider("n_s", 0.9, 1.1, 0.9665, 0.001),
            "mnu": st.slider("m_ν (eV)", 0.0, 0.5, 0.1, 0.01),
            "w0": st.slider("w₀", -1.5, 0.5, -1.0, 0.1),
            "wa": st.slider("w_a", -0.5, 0.5, 0.0, 0.1),
            "lkfid": st.slider("log₁₀(k_fid)", -3.0, 2.0, -1.0, 0.1),
            "z": st.slider("Redshift (z)", 0.0, 10.0, 0.0, 0.1)
        }
        
        vary_param = st.selectbox("Vary Parameter", ["sigma8", "Om", "Ob", "h", "ns", "mnu", "w0", "wa"],index=2)
        interpolation = st.selectbox("Interpolation", 
                                   ["linear", "spline", "hv", "vh"],
                                   index=1)

    # Cosmology calculations
    a = 1 / (1 + fiducial["z"])
    As = sigma8_to_As_max_precision(
        fiducial["sigma8"], fiducial["Om"], fiducial["Ob"], fiducial["h"],
        fiducial["ns"], fiducial["mnu"], fiducial["w0"], fiducial["wa"]
    )
    
    # Generate parameter range for varied parameter
    param_ranges = {
        "sigma8": (0.75, 0.85), "Om": (0.2, 0.7), "Ob": (0.01, 0.1), "h": (0.5, 0.8),
        "ns": (0.9, 1.1), "mnu":(0, 0.3), "w0": (-1.5, 0.5), "wa": (-0.5, 0.5)
    }
    param_vals = np.linspace(*param_ranges[vary_param], 5)
    
    # Calculate power spectra
    logk = np.linspace(-3, 2, 400)
    k = 10**logk
    mass_scale = np.log10(k_to_mass(k, fiducial["Om"]))
    
    # Create figures
    fig_ps = go.Figure()
    fig_rms = go.Figure()

    for pval in param_vals:
        params = fiducial.copy()
        params[vary_param] = pval
        As = sigma8_to_As_max_precision(params["sigma8"], params["Om"], params["Ob"], params["h"],
            params["ns"], params["mnu"], params["w0"], params["wa"])
        Pk = np.array([plin_new_emulated(
            ki, As, params["Om"], params["Ob"], params["h"],
            params["ns"], params["mnu"], params["w0"], params["wa"], a
        ) for ki in k])
        
        Δk = np.sqrt((k**3) / (2*np.pi**2) * Pk)
        
        # Add traces to figures
        name = f"{vary_param} = {pval:.2f}"
        fig_ps.add_trace(go.Scatter(
            x=logk, y=Pk, name=name,
            line=dict(shape=interpolation)
        ))
        fig_rms.add_trace(go.Scatter(
            x=logk, y=Δk, name=name,
            line=dict(shape=interpolation), showlegend=False
        ))

    # Format power spectrum plot
    fig_ps.update_layout(
        title=f"Power Spectrum at z = {fiducial['z']:.1f}",
        xaxis_title="log₁₀(k [Mpc⁻¹ h])",
        yaxis_title="P(k)",
        yaxis_type="log",
        hovermode="x unified"
    )
    
    # Format RMS plot
    fig_rms.update_layout(
        title=f"RMS Fluctuations at z = {fiducial['z']:.1f}",
        xaxis_title="log₁₀(k [Mpc⁻¹ h])",
        yaxis_title="Δ(k)",
        yaxis_type="log",
        hovermode="x unified"
    )

    # Display plots
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_ps, use_container_width=True)
    with col2:
        st.plotly_chart(fig_rms, use_container_width=True)

if __name__ == "__main__":
    main()