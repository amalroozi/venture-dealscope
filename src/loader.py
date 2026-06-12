import pandas as pd
import numpy as np

def load_data(path='data/objects.csv'):
    df = pd.read_csv(path, encoding='latin-1')
    companies = df[df['entity_type'] == 'Company'].copy()
    clean = companies.dropna(subset=['category_code', 'short_description'])
    
    clean = clean.copy()
    clean['short_description'] = clean['short_description'].str.encode('ascii', 'ignore').str.decode('ascii')
    clean['founded_year'] = pd.to_datetime(clean['founded_at'], errors='coerce').dt.year
    
    cols = ['name', 'category_code', 'short_description', 'status', 
            'founded_year', 'funding_rounds', 'funding_total_usd', 'country_code']
    
    data = clean[cols].reset_index(drop=True)
    return data