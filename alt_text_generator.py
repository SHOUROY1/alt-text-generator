"""
Alt Text Generator - Streamlit App
Generates SEO-friendly alt text for images using AI vision models.
"""

import streamlit as st
import pandas as pd
import time
from io import BytesIO

# Optional imports - user needs at least one
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def generate_alt_text_openai(image_url: str, api_key: str) -> str:
    """Generate alt text using OpenAI GPT-4o-mini."""
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an SEO expert. Generate a single, concise, descriptive sentence for image alt text. Focus on the main subject, relevant details, and natural keyword inclusion. Do not start with 'Image of' or 'Photo of'. Just describe what's in the image directly."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": "Generate a 1-sentence SEO-friendly alt text for this image."
                    }
                ]
            }
        ],
        max_tokens=100
    )
    
    return response.choices[0].message.content.strip()


def generate_alt_text_anthropic(image_url: str, api_key: str) -> str:
    """Generate alt text using Claude Haiku."""
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "Generate a 1-sentence SEO-friendly alt text for this image. Be concise and descriptive. Focus on the main subject and relevant details. Do not start with 'Image of' or 'Photo of'. Just describe what's in the image directly."
                    }
                ]
            }
        ]
    )
    
    return response.content[0].text.strip()


def process_images(df: pd.DataFrame, url_column: str, api_key: str, model_provider: str, progress_bar, status_text) -> pd.DataFrame:
    """Process all images and generate alt text."""
    alt_texts = []
    total = len(df)
    
    for idx, row in df.iterrows():
        image_url = row[url_column]
        
        status_text.text(f"Processing image {idx + 1} of {total}...")
        
        try:
            if pd.isna(image_url) or str(image_url).strip() == "":
                alt_texts.append("")
            else:
                if model_provider == "OpenAI (GPT-4o-mini)":
                    alt_text = generate_alt_text_openai(str(image_url).strip(), api_key)
                else:
                    alt_text = generate_alt_text_anthropic(str(image_url).strip(), api_key)
                alt_texts.append(alt_text)
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
                
        except Exception as e:
            alt_texts.append(f"Error: {str(e)}")
        
        progress_bar.progress((idx + 1) / total)
    
    df["Alt Text"] = alt_texts
    return df


def main():
    st.set_page_config(
        page_title="Alt Text Generator",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ AI Alt Text Generator")
    st.markdown("Upload a CSV with image URLs and generate SEO-friendly alt text descriptions using AI vision models.")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    available_models = []
    if OPENAI_AVAILABLE:
        available_models.append("OpenAI (GPT-4o-mini)")
    if ANTHROPIC_AVAILABLE:
        available_models.append("Anthropic (Claude Haiku)")
    
    if not available_models:
        st.error("❌ No AI libraries installed. Please install `openai` or `anthropic` package.")
        st.code("pip install openai anthropic", language="bash")
        return
    
    model_provider = st.sidebar.selectbox(
        "Select AI Model",
        available_models,
        help="Choose which vision model to use for generating alt text"
    )
    
    # API Key input
    if "OpenAI" in model_provider:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
    else:
        api_key = st.sidebar.text_input(
            "Anthropic API Key",
            type="password",
            help="Enter your Anthropic API key"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📋 Instructions
    1. Enter your API key
    2. Upload a CSV file with image URLs
    3. Select the column containing URLs
    4. Click 'Generate Alt Text'
    5. Download the results
    """)
    
    # Main content
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must contain a column with image URLs"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown(f"**Total rows:** {len(df)}")
            
            # Column selection
            url_column = st.selectbox(
                "Select the column containing Image URLs",
                df.columns.tolist(),
                help="Choose the column that contains the image URLs"
            )
            
            # Preview selected column
            st.markdown("**Sample URLs from selected column:**")
            sample_urls = df[url_column].dropna().head(3).tolist()
            for url in sample_urls:
                st.code(url, language=None)
            
            # Generate button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                generate_button = st.button(
                    "🚀 Generate Alt Text",
                    use_container_width=True,
                    type="primary"
                )
            
            if generate_button:
                if not api_key:
                    st.error("⚠️ Please enter your API key in the sidebar.")
                else:
                    st.subheader("⏳ Processing...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        result_df = process_images(
                            df.copy(),
                            url_column,
                            api_key,
                            model_provider,
                            progress_bar,
                            status_text
                        )
                        
                        status_text.text("✅ Processing complete!")
                        
                        st.subheader("📋 Results")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Success/error summary
                        errors = result_df["Alt Text"].str.startswith("Error:").sum()
                        success = len(result_df) - errors
                        
                        col1, col2 = st.columns(2)
                        col1.metric("✅ Successful", success)
                        col2.metric("❌ Errors", errors)
                        
                        # Download button
                        csv_buffer = BytesIO()
                        result_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download CSV with Alt Text",
                            data=csv_buffer.getvalue(),
                            file_name="images_with_alt_text.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")
                        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
    
    else:
        # Show example format
        st.info("👆 Upload a CSV file to get started")
        
        st.subheader("📝 Expected CSV Format")
        example_df = pd.DataFrame({
            "Product Name": ["Blue Sneakers", "Red Dress", "Leather Bag"],
            "Image URL": [
                "https://example.com/sneakers.jpg",
                "https://example.com/dress.jpg",
                "https://example.com/bag.jpg"
            ],
            "Price": [99.99, 149.99, 199.99]
        })
        st.dataframe(example_df, use_container_width=True)


if __name__ == "__main__":
    main()
