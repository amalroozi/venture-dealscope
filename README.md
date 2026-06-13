## Venture Dealscope


Live at [venturedealscope.streamlit.app](https://venturedealscope.streamlit.app)

Dealscope is a venture capital deal flow analyzer that discovers, clusters, and scores startups by sector to surface the most fundable opportunities.  (Saves VC hours of manual research)


.

## Overview
(Based on the free Kaggle Investments dataset from (justinas/startup-investments).

Input a sector. The tool pulls all relevant startups from a Crunchbase dataset, embeds their descriptions into semantic vectors, clusters them into sub-sectors, scores each company on traction signals, and renders an interactive landscape map. The output answers what a VC associate spends hours on manually: what does this space look like, and who are the most interesting companies in it.

## How it works

1. Company descriptions are embedded into 384-dimensional vectors using sentence-transformers (all-MiniLM-L6-v2). Companies doing similar things cluster together in this space.
2. KMeans groups the embeddings into sub-sectors. PCA reduces to 2D for visualization.
3. LLaMA 3.3 70B via Groq labels each cluster with a human-readable sub-sector name.
4. Each company is scored out of 100 on four traction signals.

## Traction Score

| Signal | Weight | Logic |
|--------|--------|-------|
| Total funding (log-scaled) | 40% | Log scaling prevents billion-dollar outliers from dominating |
| Funding rounds | 30% | More rounds = more investor validation |
| Company age | 20% | Newer companies score higher |
| Status | 10% | Operating/IPO > Acquired > Closed |

## Stack

sentence-transformers, scikit-learn, Groq API (LLaMA 3.3 70B), Streamlit, Plotly, pandas, numpy

## Data

Crunchbase startup dataset via Kaggle (justinas/startup-investments). 6,819 companies across 42 sectors after filtering for valid category and description.

## Running locally

```bash
git clone https://github.com/amalroozi/venture-dealscope.git
cd venture-dealscope
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your Groq/Other LLM API key (Groq because it has a free tier)
# place objects.csv from Kaggle into data/
streamlit run app.py
```

Download the Startup Investments dataset from Kaggle (justinas/startup-investments) and place `objects.csv` in the `data/` folder.

```bash
streamlit run app.py
```

## Data

Crunchbase startup dataset via Kaggle. 6,819 companies with category, description, funding history, and status after filtering. Not included in the repo due to licensing.
