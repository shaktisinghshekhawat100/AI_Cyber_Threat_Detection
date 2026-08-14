# AI-Based Cyber Threat Detection Framework

College-level end-to-end ML prototype for classifying network-flow records as normal traffic or common cyber attacks.

## Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Features
- Synthetic dataset generator (no external dataset required)
- Random Forest classifier
- Normal / DoS / Port Scan / Brute Force / Web Attack classes
- Streamlit dashboard
- CSV upload and batch prediction
- Threat statistics and charts
- Accuracy, precision, recall, F1 and confusion matrix when labels are present
- Downloadable prediction report

This is an academic prototype, not a production IDS.
