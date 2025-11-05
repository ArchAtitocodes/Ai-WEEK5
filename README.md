# Hospital 30-Day Readmission Prediction System

## Overview

An AI-powered clinical decision support system that predicts patient readmission risk within 30 days of hospital discharge. This project demonstrates the complete AI development workflow from problem definition through production deployment, with emphasis on ethics, fairness, and regulatory compliance.

**Assignment:** AI Development Workflow - Real-World Application  
**Institution:** Power Learn Project Academy  
**Date:** November 2025

---

## 🎯 Project Objectives

1. **Clinical Goal:** Identify patients at high risk of readmission with ≥80% sensitivity
2. **Operational Goal:** Reduce 30-day readmission rates by 20% through targeted interventions
3. **Technical Goal:** Deploy HIPAA-compliant, production-ready ML system integrated with hospital EHR

---

## 📊 Problem Statement

Hospital readmissions within 30 days of discharge are:
- A key quality metric affecting hospital reimbursement (CMS penalties)
- Costly: $17 billion annually in the US
- Often preventable with proper discharge planning and follow-up care

**Solution:** Predictive model that identifies high-risk patients at discharge, enabling care coordinators to deploy intensive interventions (home visits, medication management, early follow-ups) to those who need them most.

---

## 🏗️ Project Structure

```
readmission-prediction/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── data/
│   ├── raw/                          # Raw EHR extracts (not in repo - PHI)
│   ├── processed/                    # Preprocessed, de-identified data
│   └── synthetic_patient_data.csv    # Synthetic demo data
│
├── src/
│   ├── __init__.py
│   ├── readmission_predictor.py      # Main training pipeline
│   ├── preprocessing.py              # Data preprocessing utilities
│   ├── feature_engineering.py        # Feature creation functions
│   └── evaluation.py                 # Model evaluation tools
│
├── api/
│   ├── deployment_api.py             # Flask REST API
│   ├── requirements.txt              # API-specific dependencies
│   └── config.py                     # API configuration
│
├── monitoring/
│   ├── model_monitor.py              # Drift detection system
│   ├── fairness_audit.py             # Bias monitoring
│   └── alerting.py                   # Alert system
│
├── models/
│   ├── readmission_model.pkl         # Trained model artifact
│   ├── readmission_model_metadata.json
│   └── model_card.md                 # Model documentation
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_fairness_analysis.ipynb
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
├── docs/
│   ├── workflow_diagram.png          # AI development workflow
│   ├── architecture.md               # System architecture
│   ├── deployment_guide.md           # Deployment instructions
│   └── ethics_framework.md           # Ethical considerations
│
└── deployment/
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── kubernetes/
    │   └── deployment.yaml
    └── scripts/
        ├── deploy.sh
        └── monitor.sh
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- Virtual environment tool

### Installation

```bash
# Clone repository
git clone https://github.com/ArchAtitocodes/Ai-WEEK5.git
cd readmission-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Training Pipeline

```bash
# Train model on synthetic data
python src/readmission_predictor.py

# Expected output:
# - Trained model saved to models/readmission_model.pkl
# - Performance metrics displayed
# - Fairness audit results
```

### Start API Server

```bash
# Start Flask development server
cd api
export FLASK_ENV=development
export VALID_API_KEYS=test-key-123
python deployment_api.py

# Server runs on http://localhost:5000
```

### Test Prediction

```bash
# Example cURL request
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-key-123" \
  -d '{
    "patient_id": "patient_001",
    "features": {
      "age": 72,
      "gender_encoded": 1,
      "num_medications": 8,
      "length_of_stay": 6,
      "num_procedures": 3,
      "charlson_index": 5,
      "prior_admissions_12mo": 2,
      "emergency_admission": 1,
      "icu_stay": 1,
      "had_missing_data": 0,
      "polypharmacy": 1,
      "age_high_risk": 1,
      "complex_patient": 1,
      "frequent_flyer": 1,
      "extended_stay": 0,
      "critical_pathway": 1,
      "diagnosis_heart_failure": 1,
      "diagnosis_copd": 0,
      "diagnosis_diabetes": 1
    }
  }'
```

---

## 📈 Model Performance

### Test Set Results (Synthetic Data)

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **Recall (Sensitivity)** | 80.0% | Catches 80% of patients who will be readmitted |
| **Precision** | 41.4% | 41% of flagged patients are actual readmissions |
| **AUC-ROC** | 0.82 | Strong discriminative ability |
| **Specificity** | 80.0% | Correctly identifies 80% of non-readmissions |

### Cost-Benefit Analysis

- **False Positive Cost:** $50/patient (unnecessary follow-up)
- **True Positive Benefit:** $10,000/patient (prevented readmission)
- **Net ROI:** 200:1 benefit-cost ratio

### Feature Importance (Top 5)

1. **prior_admissions_12mo** (0.185) - History of recent admissions
2. **charlson_index** (0.142) - Comorbidity complexity
3. **age** (0.128) - Patient age
4. **num_medications** (0.115) - Polypharmacy indicator
5. **length_of_stay** (0.098) - Duration of hospitalization

---

## 🛡️ Ethics & Fairness

### Bias Mitigation Strategies

1. **Pre-Processing**
   - Stratified data collection to ensure demographic representation
   - Removed proxies for protected attributes (zip code economic indicators)
   - Balanced training set using SMOTE

2. **In-Processing**
   - Fairness constraints during training (equalized odds)
   - Adversarial debiasing to prevent protected attribute leakage

3. **Post-Processing**
   - Group-specific thresholds to achieve equal recall across demographics
   - Quarterly fairness audits with disparate impact analysis

### Fairness Audit Results

| Demographic Group | Sample Size | Recall | Precision | AUC |
|-------------------|-------------|--------|-----------|-----|
| Age <65 | 450 | 78.5% | 39.2% | 0.80 |
| Age ≥65 | 550 | 81.3% | 43.1% | 0.83 |

**Recall Gap:** 2.8% (<10% threshold) ✓ Fairness check passed

### Ethical Considerations

- **Patient Privacy:** All data de-identified, HIPAA-compliant storage
- **Transparency:** Predictions include explanation of top risk factors
- **Human Oversight:** Clinicians can override model recommendations
- **Equity:** Regular monitoring ensures no demographic group is disadvantaged

---

## 🔒 Security & Compliance

### HIPAA Compliance Measures

1. **Technical Safeguards**
   - AES-256 encryption at rest
   - TLS 1.3 for data in transit
   - Role-based access control (RBAC)
   - Multi-factor authentication (MFA)
   - Comprehensive audit logging (7-year retention)

2. **Administrative Safeguards**
   - Business Associate Agreements (BAAs) with all vendors
   - Annual security training for personnel
   - Incident response plan with 60-day breach notification
   - Annual risk assessments

3. **Physical Safeguards**
   - HIPAA-compliant data centers
   - Endpoint protection on all workstations
   - Encrypted mobile devices

### API Security

- API key authentication (rotate keys quarterly)
- Rate limiting: 200/day, 50/hour per key
- IP whitelisting for production environments
- Request/response logging for audit trails
- Input validation to prevent injection attacks

---

## 📊 Monitoring & Maintenance

### Continuous Monitoring

The system includes automated monitoring for:

1. **Performance Drift**
   - Weekly AUC, recall, precision tracking
   - Alert if metrics drop >5%
   - Automated retraining triggers

2. **Feature Drift**
   - Kolmogorov-Smirnov tests on feature distributions
   - Alert if significant distribution shifts detected
   - Track seasonal patterns (e.g., flu season)

3. **Concept Drift**
   - Compare prediction accuracy to baseline
   - Detect changes in feature-target relationships
   - Monthly retraining if drift exceeds threshold

4. **Fairness Drift**
   - Quarterly stratified performance audits
   - Track recall gaps across demographic groups
   - Alert if disparities exceed 10%

### Maintenance Schedule

- **Daily:** System health checks, uptime monitoring
- **Weekly:** Performance metric review
- **Monthly:** Drift detection analysis
- **Quarterly:** Fairness audits, model retraining
- **Annually:** Comprehensive security audit, bias review

---

## 🚢 Deployment

### Production Deployment (AWS Example)

```bash
# Build Docker container
docker build -t readmission-api:latest -f deployment/docker/Dockerfile .

# Deploy to AWS ECS
aws ecs create-service \
  --cluster hospital-ml-cluster \
  --service-name readmission-prediction \
  --task-definition readmission-api:1 \
  --desired-count 3 \
  --launch-type FARGATE

# Configure load balancer (HTTPS only)
aws elbv2 create-load-balancer \
  --name readmission-lb \
  --subnets subnet-xxx subnet-yyy \
  --security-groups sg-xxx \
  --scheme internal  # Internal-only for HIPAA
```

### Environment Variables (Production)

```bash
FLASK_ENV=production
SECRET_KEY=<secure-random-key>
VALID_API_KEYS=<comma-separated-keys>
DATABASE_URL=<encrypted-connection-string>
SENTRY_DSN=<error-tracking-url>
LOG_LEVEL=INFO
```

### Health Checks

```bash
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 30
  periodSeconds: 10
```

---

## 📚 Documentation

### Model Card

Complete model card documenting:
- Intended use cases and limitations
- Training data characteristics
- Performance metrics across demographics
- Ethical considerations
- Monitoring requirements

**Location:** `models/model_card.md`

### Architecture Documentation

System architecture diagrams showing:
- Data flow from EHR to predictions
- API integration points
- Security layers
- Monitoring infrastructure

**Location:** `docs/architecture.md`

### Workflow Diagram

Visual representation of complete AI development workflow from problem definition through deployment and monitoring.

**Location:** `docs/workflow_diagram.png`

---

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_model.py -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage Goals

- **Code Coverage:** >80%
- **Model Tests:** Prediction accuracy, bias metrics, edge cases
- **API Tests:** Authentication, rate limiting, error handling
- **Integration Tests:** End-to-end prediction pipeline

---

## 🤝 Contributing

### Development Workflow

1. Fork repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Make changes with clear commits
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Update documentation
7. Submit pull request

### Code Standards

- Follow PEP 8 style guide
- Include docstrings for all functions
- Add type hints where appropriate
- Comment complex logic
- Update requirements.txt if dependencies change

---

## 📝 Assignment Deliverables

### Part 1: Short Answer Questions ✅
- Problem definition with objectives, stakeholders, KPIs
- Data collection and preprocessing strategy
- Model selection and justification
- Evaluation metrics and deployment considerations

### Part 2: Case Study Application ✅
- Complete hospital readmission prediction system
- Data strategy with ethical considerations
- Model development with confusion matrix analysis
- Deployment plan with HIPAA compliance
- Overfitting mitigation strategies

### Part 3: Critical Thinking ✅
- Bias impact analysis with mitigation strategies
- Interpretability vs. accuracy trade-off discussion
- Resource constraint considerations

### Part 4: Reflection & Workflow ✅
- Reflection on challenging aspects
- Improvement recommendations
- Complete workflow diagram

---

## 📖 References

1. Obermeyer, Z., et al. (2019). "Dissecting racial bias in an algorithm used to manage the health of populations." *Science*, 366(6464), 447-453.

2. Rajkomar, A., et al. (2018). "Ensuring Fairness in Machine Learning to Advance Health Equity." *Annals of Internal Medicine*, 169(12), 866-872.

3. Centers for Medicare & Medicaid Services. (2023). "Hospital Readmissions Reduction Program (HRRP)."

4. U.S. Department of Health and Human Services. (2013). "HIPAA Security Rule."

5. Sendak, M. P., et al. (2020). "A Path for Translation of Machine Learning Products into Healthcare Delivery." *EMJ Innovations*, 10(1), 19-00172.

---

## 📄 License

This project is for educational purposes as part of the Power Learn Project Academy AI Engineering program.

**Note:** This is a demonstration system using synthetic data. For production deployment with real patient data, conduct comprehensive security audits, obtain IRB approval, and ensure full HIPAA compliance.

---

## 🧑‍💻 Author  

**Course:** AI for Software Engineering  
**Week:** 5  
**Institution:** [PLP AFRICA ACADEMY]  

** GROUP MEMBERS:**  

**1.** [Stephen Odhiambo]  **Email:** (stephenodhiambo008@gmail.com) 
**2.** [Jackline Biwott]  **Email:** (biwottjackline72@gmail.com) 

November 2025

---

## 🆘 Support

For questions or issues:
- Create an issue in GitHub repository
- Contact: [stephenodhiambo008@gmail.com , biwottjackline72@gmail.com]
- PLP Community 

---

## 🙏 Acknowledgments

- Power Learn Project Academy for curriculum and guidance
- Healthcare AI research community for ethical frameworks
- Open-source contributors to scikit-learn, pandas, Flask

---

**Last Updated:** November 4, 2025  
**Version:** 1.0.0
