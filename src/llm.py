from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def label_clusters(filtered_df):
    labels = {}
    for cluster_id in filtered_df['cluster'].unique():
        cluster_companies = filtered_df[filtered_df['cluster'] == cluster_id]
        sample_descriptions = cluster_companies['short_description'].head(5).tolist()
        
        prompt = f"""Here are descriptions of startups in the same cluster:
{chr(10).join(f'- {d}' for d in sample_descriptions)}
Give this cluster a concise sub-sector label (3-5 words max). Reply with only the label, nothing else."""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        labels[cluster_id] = response.choices[0].message.content.strip()
    
    return labels