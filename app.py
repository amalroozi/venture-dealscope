import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.loader import load_data
from src.embedder import get_embeddings
from src.clusterer import cluster_and_reduce
from src.scorer import compute_traction_score
from src.llm import label_clusters

st.set_page_config(page_title="Venture DealScope", layout="wide")

st.title(" Venture Dealscope")
st.caption("Hi!, Discover where investors are deploying capital and which startups are gaining momentum. Automatically cluster and rank startups by sector to accelerate deal sourcing for Venture Capitals.")

@st.cache_data
def load_and_embed():
    data = load_data()
    data['traction_score'] = compute_traction_score(data)
    embeddings = get_embeddings(data['short_description'].tolist())
    return data, embeddings

with st.spinner("Loading startup data and generating embeddings..."):
    data, embeddings = load_and_embed()

st.success(f"Loaded {len(data):,} companies across {data['category_code'].nunique()} sectors")

sector = st.text_input("Enter a sector to analyze", placeholder="e.g. biotech, software, cleantech")

if st.button("Analyze") and sector:
    mask = data['category_code'].str.contains(sector, case=False, na=False)
    filtered = data[mask].copy()
    filtered_embeddings = embeddings[mask.values]

    if len(filtered) < 10:
        st.warning(f"Only {len(filtered)} companies found for '{sector}'. Try a broader term.")
    else:
        with st.spinner(f"Analyzing {len(filtered)} companies in '{sector}'..."):
            filtered = cluster_and_reduce(filtered, filtered_embeddings)
            cluster_labels = label_clusters(filtered)
            filtered['cluster_label'] = filtered['cluster'].map(cluster_labels)
            top = filtered.sort_values('traction_score', ascending=False).head(20)

        st.subheader(f"Top 20 Companies in '{sector}'")
        st.dataframe(
            top[['name', 'cluster_label', 'traction_score', 'funding_total_usd', 'funding_rounds', 'status', 'country_code']],
            use_container_width=True
        )

        st.subheader("Sector Landscape Map")
        fig = px.scatter(
            filtered,
            x='x', y='y',
            color='cluster_label',
            hover_name='name',
            hover_data={'traction_score': True, 'funding_total_usd': True, 'x': False, 'y': False},
            size=filtered['traction_score'].clip(lower=1),
            title=f"Startup Cluster Map — {sector.title()}",
            labels={'cluster_label': 'Sub-sector'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
