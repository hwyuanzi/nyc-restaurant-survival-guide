import streamlit as st

def apply_apple_theme():
    """
    Injects ultra-modern Apple iOS / macOS CSS.
    Optimized for Streamlit Performance (Removed expensive backdrop-filters).
    """
    
    st.markdown("""
        <style>
        /* 1. Global Typography: San Francisco / Apple Native */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            -webkit-font-smoothing: antialiased;
        }

        /* 2. Hide noisy default headers and footers */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {background-color: transparent !important;}

        /* 3. Streamlined Clean Background (Hardware Accelerated) */
        .stApp {
            background: #f4f6fa !important;
        }

        /* 4. Clean macOS Card Main Container (Optimized for 60FPS Interaction) */
        .block-container {
            background: #FFFFFF !important;
            border-radius: 24px !important;
            border: 1px solid rgba(0, 0, 0, 0.05) !important;
            padding: 3rem 4rem 4rem 4rem !important;
            margin-top: 2rem !important;
            margin-bottom: 2rem !important;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.04) !important;
            max-width: 95% !important;  /* Force Wide Mode */
        }

        /* 5. Minimalist Sidebar & Pro Navigation Tabs */
        [data-testid="stSidebar"] {
            background-color: #F8F9FA !important;
            border-right: 1px solid rgba(0,0,0,0.05) !important;
            box-shadow: 2px 0 20px rgba(0,0,0,0.02) !important;
        }

        /* Sidebar Tab Base Styling for Apple Native Look */
        section[data-testid="stSidebar"] nav > ul > li > a {
            padding: 0.65rem 1rem !important;
            margin: 0.3rem 1.2rem !important;
            border-radius: 10px !important;
            transition: all 0.2s ease !important;
            font-size: 15px !important;
            font-weight: 500 !important;
            color: #1D1D1F !important;
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
        }

        /* Hover effect */
        section[data-testid="stSidebar"] nav > ul > li > a:hover {
            background-color: rgba(0, 122, 255, 0.06) !important;
            color: #007AFF !important;
        }

        /* Active Selected Tab effect */
        section[data-testid="stSidebar"] nav > ul > li > a[aria-current="page"] {
            background-color: #007AFF !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 10px rgba(0, 122, 255, 0.2) !important;
        }
        
        /* Hide SVG icon in sidebar since we rely on Emojis */
        section[data-testid="stSidebar"] nav > ul > li > a > svg {
            display: none !important;
        }

        /* 6. Liquid iOS Buttons (Kept the Apple Button magic) */
        div.stButton > button {
            background: linear-gradient(135deg, #007AFF 0%, #0056b3 100%) !important;
            color: #FFFFFF !important;
            border-radius: 12px !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px !important;
            box-shadow: 0 4px 14px rgba(0, 122, 255, 0.25) !important;
            transition: all 0.2s ease !important;
        }
        
        div.stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(0, 122, 255, 0.35) !important;
        }
        div.stButton > button:active {
            transform: translateY(1px) scale(0.98) !important;
        }

        /* 7. Inputs & Select Boxes (Soft Apple fields) */
        .stTextInput>div>div>input, .stSelectbox>div>div>div, .stMultiSelect>div>div>div {
            border-radius: 10px !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
            background-color: #FAFAFC !important;
            color: #1D1D1F !important;
            transition: all 0.2s ease !important;
        }
        
        .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
            border-color: #007AFF !important;
            box-shadow: 0 0 0 3px rgba(0,122,255,0.15) !important;
            background-color: #FFFFFF !important;
        }

        /* 8. Text Formatting */
        h1 {
            color: #1D1D1F !important;
            font-weight: 800 !important;
            letter-spacing: -1.0px !important;
            margin-bottom: 1.5rem !important;
        }
        
        h2, h3 {
            color: #1D1D1F !important;
            font-weight: 700 !important;
            letter-spacing: -0.8px !important;
        }

        /* 9. Alerts (iOS Notification style, optimized) */
        div[data-testid="stAlert"] {
            border-radius: 14px !important;
            border: 1px solid rgba(0,0,0,0.05) !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.04) !important;
            background: #FFFFFF !important;
            color: #1D1D1F !important;
            padding: 1.0rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
