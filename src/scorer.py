import pandas as pd
import numpy as np

def compute_traction_score(df):
    scores = pd.DataFrame(index=df.index)
    
    log_funding = np.log1p(df['funding_total_usd'])
    scores['funding_score'] = (log_funding - log_funding.min()) / (log_funding.max() - log_funding.min())
    
    scores['rounds_score'] = (df['funding_rounds'] - df['funding_rounds'].min()) / (df['funding_rounds'].max() - df['funding_rounds'].min())
    
    scores['age_score'] = df['founded_year'].apply(lambda x: min((x - 1990) / 25, 1) if pd.notnull(x) else 0.5)
    
    status_map = {'operating': 1.0, 'acquired': 0.8, 'ipo': 1.0, 'closed': 0.0}
    scores['status_score'] = df['status'].map(status_map).fillna(0.5)
    
    final = (
        0.4 * scores['funding_score'] +
        0.3 * scores['rounds_score'] +
        0.2 * scores['age_score'] +
        0.1 * scores['status_score']
    )
    return (final * 100).round(1)