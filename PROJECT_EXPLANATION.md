# Project Explanation

## Problem
Manual analysis of network traffic is difficult at scale. This system uses machine learning to classify network-flow records.

## Objectives
- Detect suspicious network behavior
- Classify common threat categories
- Show results through a dashboard
- Provide confidence and downloadable reports

## Threat Classes
Normal, DoS, Port Scan, Brute Force, Web Attack.

## Algorithm
Random Forest Classifier with one-hot encoding for the protocol feature.

## Workflow
Data generation/collection -> preprocessing -> train/test split -> Random Forest -> evaluation -> saved model -> Streamlit prediction -> visualization.

## Future Scope
Real-time packet capture, live monitoring, IDS/SIEM integration, deep learning, explainable AI and alerting.

## Limitation
The included dataset is synthetic for academic demonstration. A production security system needs validated real-world data and extensive testing.
