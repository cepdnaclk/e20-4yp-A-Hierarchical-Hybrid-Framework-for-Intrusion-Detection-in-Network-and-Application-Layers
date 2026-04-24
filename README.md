# Hierarchical Hybrid Framework for Intrusion Detection in Network and Application Layers

A multi-layered intrusion detection system that processes network and application traffic in parallel using a combination of supervised and threshold-based machine learning techniques. The framework uses a three-layer cascade (anomaly detection, known/unknown classification, and multi-class attack identification) to detect both known cyber-attacks and potential zero-day threats in real time.

Built and evaluated on the CSE-CIC-IDS2018 dataset with 41 selected features across network and application layers.

## Problem

Traditional intrusion detection systems process network and application layers sequentially or in isolation. This allows sophisticated application-layer attacks (SQL Injection, XSS, Command Injection) to evade detection because they appear normal at the network level. Supervised models perform well on known threats but fail against zero-day attacks, while unsupervised methods suffer from high false positive rates. There is a clear need for a framework that handles both layers in parallel with intelligent fusion and reliable unknown threat detection.

## Approach

The framework operates as a fully parallel dual-stream architecture with two independent processing pipelines for network-layer (22 features) and application-layer (19 features) traffic. Each stream passes through three cascading layers:

**Layer 1 - Anomaly Detector (Binary Classifier)**
Distinguishes benign traffic from attacks. LightGBM is used for the Network Stream and XGBoost for the Application Stream. An initial unsupervised approach using Isolation Forest, Autoencoder, and One-Class SVM was tested first but abandoned due to extremely poor results (accuracy as low as 3%). Switching to supervised classification reduced the false-negative rate from over 95% down to under 6%.

**Layer 2 - Known/Unknown Identifier**
A zero-overhead threshold-based decision mechanism that uses the confidence scores from Layer 1. Flows with high confidence (>= 0.7) are routed as "known attacks" to Layer 3 for classification. Flows with low confidence are flagged as potential zero-day threats and escalated immediately for administrator review. This replaced unsupervised models (Isolation Forest, One-Class SVM) that had critically low precision (0.26 to 0.27), raising precision to 0.84 and F1-score to 0.87.

**Layer 3 - Multi-Class Classifier**
XGBoost classifies known attacks into 14 distinct attack families. The macro F1-score (0.92 on the Network Stream) ensures minority attack classes like Slowloris and SQL Injection receive fair treatment alongside dominant classes like DDoS.

**Meta-Learner Fusion**
Results from both network and application streams are combined through a meta-learner that resolves conflicting decisions and produces final alerts.

## Feature Selection

41 features were selected using a rank aggregation of four methods:
- Correlation Coefficients (eliminating redundancy above 0.9)
- Mutual Information (capturing non-linear dependencies)
- Random Forest feature importance
- Recursive Feature Elimination with SVM

An initial 20-feature set resulted in poor performance (F1 = 0.08). Expanding to 41 features improved performance by 10 to 15% and became the foundation for all model training.

## Dataset

The CSE-CIC-IDS2018 dataset was chosen for its breadth of both network-level and application-level features (80+ total), coverage of 16 attack types, and relevance to modern enterprise traffic. The data was intentionally balanced to 60% benign and 40% attack flows, resulting in 4,122,352 benign records and 2,748,235 attack records. Preprocessing included duplicate removal, median imputation for missing/infinite values, min-max normalization, and one-hot encoding for categorical fields.

## Key Results

| Metric | Value |
|---|---|
| Overall Pipeline Accuracy | 97.8% |
| Macro F1-Score | 0.91 |
| Network Layer 1 Accuracy | 98% |
| Network Layer 3 Accuracy | 97% |
| Application Layer 1 Accuracy | 84.55% |
| Application Layer 3 Accuracy | 74.23% |
| Layer 2 Unknown F1-Score | 0.87 |
| Layer 2 False Alarm Rate | < 5% |

### Real-Time Evaluation (Attacker vs Victim VM Setup)

| Metric | Value |
|---|---|
| Known Attack Detection | 97% |
| Zero-Day Recall | 91% |
| Avg Latency per Flow | 68ms |
| False Positive Rate | < 4% |
| Throughput | ~8,500 req/s |
| CPU Usage at Peak | 35% |
| Memory Usage at Peak | 42% |

Real-time tests used Wireshark captures with CICFlowMeter for feature extraction. Attacks were generated using standard tools including Slowloris, LOIC/HOIC, SQLMap, and Hydra across all 14 attack families.

## Limitations

- The application stream lags behind the network stream significantly (~84% vs 98% at Layer 1, ~74% vs 97% at Layer 3)
- Layer 2 precision for the application stream's unknown class drops to near-zero due to extreme class imbalance (only 17 unknown cases among 500K+ known ones)
- The 0.7 confidence threshold was optimized for CSE-CIC-IDS2018 and may need tuning for other network environments
- Detection performance drops on rare attack types like web-based brute force and infiltration
- No active/continuous learning mechanism; the system depends on manual retraining with fresh data

## Team

- **E/19/174** - Jegatheesan S.
- **E/20/099** - Eniyavan T.
- **E/20/416** - Vithushan E.T.L.

**Supervisor:** Mr. Biswajith A.K. Dissanayake

Department of Computer Engineering, University of Peradeniya

## Links

- [Project Page](https://cepdnaclk.github.io/e20-4yp-Hierarchical-Hybrid-Framework-for-Intrusion-Detection)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)
