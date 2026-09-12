import streamlit as st
import google.generativeai as genai
import pandas as pd
import json
import requests
from PIL import Image
import time
import io

# --- Configure the Page ---
st.set_page_config(page_title="Product SKU Categorizer", layout="wide")
st.title("📦 AI Product SKU Categorizer (Images & Text)")

# --- API Key Setup ---
api_key = st.secrets.get("GEMINI_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("Please add your GEMINI_API_KEY to the Streamlit secrets.")

# --- Load Taxonomy from Background ---
@st.cache_data
def load_taxonomy():
    try:
        tax_df = pd.read_excel("taxonomy for gemini.xlsx")
        
        fam_col, type_col, sub_col = tax_df.columns[0], tax_df.columns[1], tax_df.columns[2]
        
        taxonomy_dict = {}
        for family, group in tax_df.groupby(fam_col):
            taxonomy_dict[str(family)] = {}
            for p_type, sub_group in group.groupby(type_col):
                subtypes = sub_group[sub_col].dropna().astype(str).unique().tolist()
                taxonomy_dict[str(family)][str(p_type)] = subtypes
                
        return taxonomy_dict, sorted(list(taxonomy_dict.keys()))
    except Exception as e:
        st.error(f"Error reading Taxonomy file. Ensure 'taxonomy for gemini.xlsx' is in your GitHub repo! Error: {e}")
        return {}, []

# --- Helper Function to Display and Download ---
def display_and_download(results_list):
    if results_list:
        df = pd.DataFrame(results_list)
        
        expected_columns = [
            "Source", "Title_EN", "Title_AR", "Family", "Type", "Subtype", 
            "Color_Family", "Color_Name", "Brand", "Size", 
            "Description_EN", "Description_AR", 
            "Feature_Bullet_1_EN", "Feature_Bullet_1_AR", 
            "Feature_Bullet_2_EN", "Feature_Bullet_2_AR", 
            "Feature_Bullet_3_EN", "Feature_Bullet_3_AR"
        ]
        columns_to_use = [col for col in expected_columns if col in df.columns]
        df = df[columns_to_use]
        
        st.success("Processing Complete!")
        st.dataframe(df, use_container_width=True)
        
        csv_export = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Download Data as CSV",
            data=csv_export,
            file_name='processed_skus.csv',
            mime='text/csv',
        )

# --- App UI & Logic ---
taxonomy_dict, family_list = load_taxonomy()

if not family_list:
    st.stop()

st.write("### 1. Select the Product Family")
selected_family = st.selectbox("Choose a Family from your Taxonomy:", options=family_list)

allowed_types_subtypes = taxonomy_dict[selected_family]
allowed_json_string = json.dumps(allowed_types_subtypes, indent=2)

st.write("### 2. Choose Your Input Method")
st.write(f"*(The AI will ONLY search for Types and Subtypes within **{selected_family}**)*")

# --- PROMPTS ---
BASE_RULES = f"""
The user has designated that this product belongs to the Family: "{selected_family}".
CRITICAL: You must categorize the product's Type and Subtype using ONLY the allowed list below for this specific family.
If the product does absolutely not fit ANY of the Types or Subtypes in this list, you MUST output "Not Found" for both Type and Subtype.

Allowed Types and Subtypes for "{selected_family}" (JSON Format - Type -> [Subtypes]):
{allowed_json_string}

Return a SINGLE JSON object with EXACTLY the following keys:
- "Title_EN": Short title of the product in English
- "Title_AR": Short title of the product translated into Arabic
- "Family": "{selected_family}"
- "Type": Must strictly match a Type key from the allowed list above. If none match, output "Not Found".
- "Subtype": Must strictly match a Subtype string nested under the chosen Type. If none match, output "Not Found".
- "Color_Family": Broad color category (e.g., Red, Blue). If not found, output "Not Found"
- "Color_Name": Specific color shade. If not found, output "Not Found"
- "Brand": Brand name if clearly visible. If not found, output "Not Found"
- "Size": Size or dimensions if clearly visible. If not found, output "Not Found"
- "Description_EN": Powerful, catchy one-paragraph e-commerce description in English
- "Description_AR": Natural, highly engaging Arabic translation of the description
- "Feature_Bullet_1_EN": Key feature/benefit 1 in English
- "Feature_Bullet_1_AR": Arabic translation of feature 1
- "Feature_Bullet_2_EN": Key feature/benefit 2 in English
- "Feature_Bullet_2_AR": Arabic translation of feature 2
- "Feature_Bullet_3_EN": Key feature/benefit 3 in English
- "Feature_Bullet_3_AR": Arabic translation of feature 3
"""

DYNAMIC_IMAGE_PROMPT = f"You are an expert inventory categorizer. Analyze the provided product image. First, determine an appropriate short title in English, and translate it to Arabic.\n{BASE_RULES}"
DYNAMIC_TEXT_PROMPT = f"You are an expert inventory categorizer. Analyze the provided product title/text. Polish the title for e-commerce in English, and translate it to Arabic.\n{BASE_RULES}"
DYNAMIC_COMBINED_PROMPT = f"You are an expert inventory categorizer. Analyze BOTH the provided product image and the provided text title. Combine context from both to perfectly categorize the product, identify the brand/size/color, polish the title in English, and translate it to Arabic.\n{BASE_RULES}"


# --- TABS SETUP ---
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Images", "🔗 Paste URLs", "📝 Paste Titles", "🖼️ + 📝 Batch (CSV/Excel)"])
model = genai.GenerativeModel('gemini-3.5-flash-lite') if api_key else None

# --- TAB 1: Direct File Upload ---
with tab1:
    uploaded_files = st.file_uploader("Upload images (PNG, JPG, JPEG, WEBP)", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)
    if st.button("Process Uploaded Images"):
        if not uploaded_files: st.warning("Upload at least one image.")
        elif not model: st.error("API Key missing.")
        else:
            results = []
            progress_bar = st.progress(0)
            status = st.empty()
            for idx, uf in enumerate(uploaded_files):
                status.text(f"Processing image {idx + 1}/{len(uploaded_files)}: {uf.name}...")
                try:
                    img = Image.open(uf)
                    resp = model.generate_content([DYNAMIC_IMAGE_PROMPT, img], generation_config={"response_mime_type": "application/json"})
                    data = json.loads(resp.text.strip())
                    data["Source"] = uf.name
                    results.append(data)
                except Exception as e: st.error(f"Failed {uf.name}: {e}")
                progress_bar.progress((idx + 1) / len(uploaded_files))
                if idx < len(uploaded_files) - 1: time.sleep(4.1)
            status.text("Finished!")
            display_and_download(results)

# --- TAB 2: Image URLs ---
with tab2:
    url_input = st.text_area("Enter Image URLs (one per line):", height=150)
    if st.button("Process Image URLs"):
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]
        if not urls: st.warning("Enter at least one URL.")
        elif not model: st.error("API Key missing.")
        else:
            results = []
            progress_bar = st.progress(0)
            status = st.empty()
            for idx, url in enumerate(urls):
                status.text(f"Processing URL {idx + 1}/{len(urls)}...")
                try:
                    res = requests.get(url, stream=True, timeout=10)
                    res.raise_for_status()
                    img = Image.open(res.raw)
                    resp = model.generate_content([DYNAMIC_IMAGE_PROMPT, img], generation_config={"response_mime_type": "application/json"})
                    data = json.loads(resp.text.strip())
                    data["Source"] = url
                    results.append(data)
                except Exception as e: st.error(f"Failed URL {url}: {e}")
                progress_bar.progress((idx + 1) / len(urls))
                if idx < len(urls) - 1: time.sleep(4.1)
            status.text("Finished!")
            display_and_download(results)

# --- TAB 3: Text Titles Only ---
with tab3:
    title_input = st.text_area("Enter Product Titles/SKUs (one per line):", height=150)
    if st.button("Process Titles"):
        titles = [t.strip() for t in title_input.split('\n') if t.strip()]
        if not titles: st.warning("Enter at least one title.")
        elif not model: st.error("API Key missing.")
        else:
            results = []
            progress_bar = st.progress(0)
            status = st.empty()
            for idx, title in enumerate(titles):
                status.text(f"Processing Title {idx + 1}/{len(titles)}...")
                try:
                    resp = model.generate_content([DYNAMIC_TEXT_PROMPT, f"Product Title: {title}"], generation_config={"response_mime_type": "application/json"})
                    data = json.loads(resp.text.strip())
                    data["Source"] = title 
                    results.append(data)
                except Exception as e: st.error(f"Failed Title {title}: {e}")
                progress_bar.progress((idx + 1) / len(titles))
                if idx < len(titles) - 1: time.sleep(4.1)
            status.text("Finished!")
            display_and_download(results)

# --- TAB 4: Combined Title & Image URL from Spreadsheet ---
with tab4:
    st.write("Upload a spreadsheet (CSV or Excel) containing a column for Titles and a column for Image URLs.")
    batch_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    if batch_file:
        # Read the file
        try:
            if batch_file.name.endswith('.csv'):
                df_input = pd.read_csv(batch_file)
            else:
                df_input = pd.read_excel(batch_file)
                
            st.success("File uploaded successfully!")
            
            # Let user map the columns
            col1, col2 = st.columns(2)
            with col1:
                title_col = st.selectbox("Select the column containing Product Titles:", options=df_input.columns)
            with col2:
                url_col = st.selectbox("Select the column containing Image URLs:", options=df_input.columns)
                
            if st.button("Process Batch (Titles + Images)"):
                if not model: st.error("API Key missing.")
                else:
                    results = []
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    total_rows = len(df_input)
                    for idx, row in df_input.iterrows():
                        current_title = str(row[title_col])
                        current_url = str(row[url_col])
                        
                        status.text(f"Processing row {idx + 1}/{total_rows}...")
                        try:
                            # 1. Fetch the image
                            res = requests.get(current_url, stream=True, timeout=10)
                            res.raise_for_status()
                            img = Image.open(res.raw)
                            
                            # 2. Pass BOTH the image and the title to the AI
                            resp = model.generate_content(
                                [DYNAMIC_COMBINED_PROMPT, f"Original Title: {current_title}", img], 
                                generation_config={"response_mime_type": "application/json"}
                            )
                            
                            # 3. Parse and save
                            data = json.loads(resp.text.strip())
                            data["Source"] = current_title # Save original title as the source reference
                            results.append(data)
                        except Exception as e:
                            st.error(f"Failed on row {idx + 1} (URL: {current_url}): {e}")
                            
                        progress_bar.progress((idx + 1) / total_rows)
                        if idx < total_rows - 1: time.sleep(4.1)
                        
                    status.text("Finished processing batch file!")
                    display_and_download(results)
                    
        except Exception as e:
            st.error(f"Error reading file: {e}")
