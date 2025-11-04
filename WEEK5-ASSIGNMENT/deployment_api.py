"""
Hospital Readmission Prediction API
Flask REST API for production deployment

This API provides secure endpoints for real-time readmission risk predictions
integrated with hospital EHR systems via HL7/FHIR protocols.

Author: AI Engineer
Date: November 2025
"""

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import hashlib
import os
from functools import wraps

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Rate limiting to prevent abuse (HIPAA security requirement)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configure logging for audit trail (HIPAA requirement)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load model at startup
MODEL_PATH = 'models/readmission_model.pkl'
model_package = None

try:
    model_package = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    logger.info(f"Model timestamp: {model_package.get('timestamp', 'Unknown')}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


def require_api_key(f):
    """
    Decorator to enforce API key authentication.
    
    In production, integrate with hospital's identity management system.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
            return jsonify({'error': 'API key required'}), 401
        
        # In production, validate against secure key store
        # This is a simplified example
        valid_keys = os.environ.get('VALID_API_KEYS', '').split(',')
        
        if api_key not in valid_keys:
            logger.warning(f"Invalid API key from {request.remote_addr}")
            return jsonify({'error': 'Invalid API key'}), 403
        
        return f(*args, **kwargs)
    
    return decorated_function


def log_prediction(patient_id, prediction, user_id=None):
    """
    Log prediction for audit trail (HIPAA compliance).
    
    Args:
        patient_id: Hashed patient identifier
        prediction: Risk prediction result
        user_id: Healthcare provider ID making request
    """
    # Hash patient_id for privacy (store hash, not actual ID)
    patient_hash = hashlib.sha256(str(patient_id).encode()).hexdigest()[:16]
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'patient_hash': patient_hash,
        'risk_score': prediction.get('risk_score'),
        'risk_level': prediction.get('risk_level'),
        'user_id': user_id,
        'ip_address': request.remote_addr
    }
    
    logger.info(f"Prediction logged: {log_entry}")
    
    # In production, store in secure database with encryption at rest
    return log_entry


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring system availability.
    
    Returns:
        200: System healthy
        503: System unavailable
    """
    try:
        # Verify model is loaded
        if model_package is None:
            return jsonify({
                'status': 'unhealthy',
                'message': 'Model not loaded'
            }), 503
        
        # Verify essential components
        required_keys = ['model', 'scaler', 'feature_names']
        missing = [k for k in required_keys if k not in model_package]
        
        if missing:
            return jsonify({
                'status': 'unhealthy',
                'message': f'Missing components: {missing}'
            }), 503
        
        return jsonify({
            'status': 'healthy',
            'model_version': model_package.get('timestamp', 'Unknown'),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'message': str(e)
        }), 503


@app.route('/predict', methods=['POST'])
@require_api_key
@limiter.limit("30 per minute")
def predict_readmission():
    """
    Generate readmission risk prediction for a patient.
    
    Request body (JSON):
    {
        "patient_id": "encrypted_patient_id",
        "features": {
            "age": 72,
            "gender_encoded": 1,
            "num_medications": 8,
            ... (all required features)
        },
        "user_id": "provider_id_123"  (optional)
    }
    
    Returns:
        200: Successful prediction
        400: Invalid request
        500: Server error
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        patient_id = data.get('patient_id')
        features = data.get('features')
        user_id = data.get('user_id')
        
        if not patient_id or not features:
            return jsonify({'error': 'patient_id and features required'}), 400
        
        # Validate features
        required_features = model_package['feature_names']
        missing_features = [f for f in required_features if f not in features]
        
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing': missing_features
            }), 400
        
        # Prepare data for prediction
        feature_df = pd.DataFrame([features])
        feature_df = feature_df[required_features]  # Ensure correct order
        
        # Scale features
        X_scaled = model_package['scaler'].transform(feature_df)
        
        # Generate prediction
        risk_proba = model_package['model'].predict_proba(X_scaled)[:, 1][0]
        risk_binary = int(risk_proba >= 0.5)
        
        # Determine risk level
        if risk_proba >= 0.7:
            risk_level = 'HIGH'
            urgency = 'IMMEDIATE'
        elif risk_proba >= 0.5:
            risk_level = 'HIGH'
            urgency = 'URGENT'
        elif risk_proba >= 0.3:
            risk_level = 'MEDIUM'
            urgency = 'ROUTINE'
        else:
            risk_level = 'LOW'
            urgency = 'STANDARD'
        
        # Get feature importance for explanation
        feature_importance = dict(zip(
            required_features,
            model_package['model'].feature_importances_
        ))
        
        # Calculate feature contributions
        top_risk_factors = []
        for feature in sorted(feature_importance, key=feature_importance.get, reverse=True)[:5]:
            if features[feature] != 0:  # Only show active risk factors
                top_risk_factors.append({
                    'feature': feature,
                    'value': features[feature],
                    'importance': float(feature_importance[feature])
                })
        
        # Generate clinical recommendation
        recommendations = []
        if risk_level in ['HIGH', 'MEDIUM']:
            recommendations.append("Assign care coordinator for discharge planning")
        if risk_proba >= 0.5:
            recommendations.append("Schedule follow-up appointment within 7 days")
            recommendations.append("Conduct medication reconciliation")
        if risk_proba >= 0.7:
            recommendations.append("Consider home visit within 48 hours")
        if features.get('polypharmacy', 0) == 1:
            recommendations.append("Pharmacy consultation for medication management")
        
        # Construct response
        prediction = {
            'patient_id': patient_id,
            'risk_score': round(float(risk_proba), 3),
            'risk_level': risk_level,
            'urgency': urgency,
            'prediction': risk_binary,
            'confidence': {
                'lower_bound': round(max(0, risk_proba - 0.05), 3),
                'upper_bound': round(min(1, risk_proba + 0.05), 3)
            },
            'top_risk_factors': top_risk_factors,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'model_version': model_package.get('timestamp', 'Unknown')
        }
        
        # Log prediction for audit
        log_prediction(patient_id, prediction, user_id)
        
        return jsonify(prediction), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': 'Unable to generate prediction'
        }), 500


@app.route('/predict/batch', methods=['POST'])
@require_api_key
@limiter.limit("10 per hour")
def predict_batch():
    """
    Batch prediction for multiple patients (e.g., daily discharge census).
    
    Request body (JSON):
    {
        "patients": [
            {
                "patient_id": "...",
                "features": {...}
            },
            ...
        ],
        "user_id": "provider_id_123"
    }
    
    Returns:
        200: Successful predictions
        400: Invalid request
    """
    try:
        data = request.get_json()
        patients = data.get('patients', [])
        user_id = data.get('user_id')
        
        if not patients:
            return jsonify({'error': 'No patients provided'}), 400
        
        if len(patients) > 100:
            return jsonify({'error': 'Batch size limited to 100 patients'}), 400
        
        predictions = []
        errors = []
        
        for idx, patient in enumerate(patients):
            try:
                # Create individual prediction request
                patient_request = {
                    'patient_id': patient.get('patient_id'),
                    'features': patient.get('features'),
                    'user_id': user_id
                }
                
                # Reuse single prediction logic
                # (In production, optimize with vectorized operations)
                with app.test_request_context(
                    json=patient_request,
                    headers={'X-API-Key': request.headers.get('X-API-Key')}
                ):
                    response = predict_readmission()
                    
                if response[1] == 200:
                    predictions.append(response[0].get_json())
                else:
                    errors.append({
                        'patient_index': idx,
                        'patient_id': patient.get('patient_id'),
                        'error': response[0].get_json()
                    })
                    
            except Exception as e:
                errors.append({
                    'patient_index': idx,
                    'patient_id': patient.get('patient_id'),
                    'error': str(e)
                })
        
        return jsonify({
            'predictions': predictions,
            'errors': errors,
            'summary': {
                'total_requested': len(patients),
                'successful': len(predictions),
                'failed': len(errors),
                'high_risk_count': sum(1 for p in predictions if p['risk_level'] == 'HIGH')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/model/info', methods=['GET'])
@require_api_key
def model_info():
    """
    Get model metadata and performance metrics.
    
    Returns model version, features, and validation performance.
    """
    try:
        info = {
            'model_version': model_package.get('timestamp', 'Unknown'),
            'features': model_package.get('feature_names', []),
            'performance_metrics': model_package.get('performance_metrics', {}),
            'last_updated': model_package.get('timestamp', 'Unknown')
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': 'Unable to retrieve model info'}), 500


@app.route('/feedback', methods=['POST'])
@require_api_key
def submit_feedback():
    """
    Submit feedback on prediction accuracy (for model improvement).
    
    Request body:
    {
        "patient_id": "...",
        "prediction_timestamp": "...",
        "actual_outcome": 0 or 1,
        "notes": "..."
    }
    """
    try:
        data = request.get_json()
        
        required_fields = ['patient_id', 'prediction_timestamp', 'actual_outcome']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Hash patient ID for privacy
        patient_hash = hashlib.sha256(
            str(data['patient_id']).encode()
        ).hexdigest()[:16]
        
        feedback_entry = {
            'patient_hash': patient_hash,
            'prediction_timestamp': data['prediction_timestamp'],
            'actual_outcome': data['actual_outcome'],
            'notes': data.get('notes', ''),
            'feedback_timestamp': datetime.now().isoformat(),
            'user_id': data.get('user_id')
        }
        
        logger.info(f"Feedback received: {feedback_entry}")
        
        # In production: Store in database for model retraining
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback recorded for model improvement'
        }), 200
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        return jsonify({'error': 'Unable to process feedback'}), 500


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded."""
    logger.warning(f"Rate limit exceeded from {request.remote_addr}")
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please try again later.'
    }), 429


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal error: {e}", exc_info=True)
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    # Development server (use Gunicorn/uWSGI in production)
    # Production: gunicorn -w 4 -b 0.0.0.0:5000 deployment_api:app
    
    # Enable HTTPS in production
    # Use environment variables for configuration
    
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_ENV') == 'development'
    )