from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

app = Flask(__name__)

# Define constants
CONCRETE_GRADES = ['M20', 'M25', 'M30', 'M35', 'M40', 'M45', 'M50']
SUPPORT_TYPES = ['One Side Fixed', 'Both Side Fixed', 'Cantilever', 'Simply Supported']
ENV_CONDITIONS = ['Normal', 'High Wind Zone', 'Heavy Rainfall', 'Coastal Area', 'Industrial Area', 'Extreme Temperature', 'High Humidity']
SEISMIC_LOAD_MIN = 0.16
SEISMIC_LOAD_MAX = 0.36

MODEL_PATH = os.path.join('models', 'structural_health_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

# Generate synthetic data for training
def generate_training_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate features
    seismic_load_enabled = np.random.choice([0, 1], size=n_samples)
    seismic_loads = np.where(seismic_load_enabled == 1, 
                           np.random.uniform(SEISMIC_LOAD_MIN, SEISMIC_LOAD_MAX, size=n_samples),
                           SEISMIC_LOAD_MIN)
    concrete_grades = np.random.choice(range(len(CONCRETE_GRADES)), size=n_samples)
    support_types = np.random.choice(range(len(SUPPORT_TYPES)), size=n_samples)
    env_conditions = np.random.choice(range(len(ENV_CONDITIONS)), size=n_samples)
    
    # Additional numerical features
    elevation = np.random.uniform(3, 100, size=n_samples)  # elevation in meters
    applied_load = np.random.uniform(10, 1000, size=n_samples)  # load in KN
    
    # Create a dataframe
    data = pd.DataFrame({
        'seismic_load_enabled': seismic_load_enabled,
        'seismic_load': seismic_loads,
        'concrete_grade': concrete_grades,
        'elevation': elevation,
        'applied_load': applied_load,
        'support_type': support_types,
        'env_condition': env_conditions
    })
    
    # Generate target (structural health) based on features
    health = (
        (np.where(seismic_load_enabled == 1, 
                 ((1 - (seismic_loads - SEISMIC_LOAD_MIN) / (SEISMIC_LOAD_MAX - SEISMIC_LOAD_MIN)) * 20),
                 20)) +  # 20 points for seismic (less is better when enabled)
        (concrete_grades / len(CONCRETE_GRADES) * 30) +  # 0-30 points for concrete
        (np.where(support_types == 0, 10, np.where(support_types == 1, 15, np.where(support_types == 2, 5, 8)))) +  # 5-15 points for support
        (np.where(env_conditions == 0, 20, np.where(env_conditions == 3, 5, np.where(env_conditions == 4, 5, 10)))) +  # 5-20 points for environment
        (np.random.normal(0, 5, n_samples))  # Random noise
    )
    
    # Normalize health to 0-100 scale
    health = np.clip(health, 0, 100)
    
    data['health_score'] = health
    
    return data

# Train or load model
def get_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        return model
    else:
        # Create directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Generate and prepare data
        data = generate_training_data(1000)
        
        X = data.drop('health_score', axis=1)
        y = data['health_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define preprocessing for categorical features
        categorical_features = ['concrete_grade', 'support_type', 'env_condition']
        numerical_features = ['seismic_load', 'elevation', 'applied_load']
        
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numerical_transformer = StandardScaler()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Create and train the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        model.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(model, MODEL_PATH)
        
        return model

# Routes
@app.route('/')
def index():
    return render_template('index.html',
                          concrete_grades=CONCRETE_GRADES,
                          support_types=SUPPORT_TYPES,
                          env_conditions=ENV_CONDITIONS)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.json
    
    # Convert to appropriate format
    seismic_load_enabled = int(data['seismic_load_enabled'])
    seismic_load = float(data['seismic_load'])
    concrete_grade = CONCRETE_GRADES.index(data['concrete_grade'])
    elevation = float(data.get('elevation', 3))
    applied_load = float(data.get('applied_load', 24))
    support_type = SUPPORT_TYPES.index(data['support_type'])
    env_condition = ENV_CONDITIONS.index(data['env_condition'])
    
    # Prepare input for model
    input_data = pd.DataFrame({
        'seismic_load_enabled': [seismic_load_enabled],
        'seismic_load': [seismic_load],
        'concrete_grade': [concrete_grade],
        'elevation': [elevation],
        'applied_load': [applied_load],
        'support_type': [support_type],
        'env_condition': [env_condition]
    })
    
    # Get prediction
    model = get_model()
    health_score = model.predict(input_data)[0]
    
    # Generate risk factors based on input parameters
    risk_factors = []
    if seismic_load_enabled and seismic_load > 0.3:
        risk_factors.append("High seismic load detected - Consider structural reinforcement")
    if concrete_grade < 2:  # Below M30
        risk_factors.append("Low concrete grade may affect structural integrity")
    if elevation > 100:
        risk_factors.append("High elevation increases wind load and seismic vulnerability")
    if applied_load > 500:
        risk_factors.append("High applied load may exceed design capacity")
    if support_type == 2:  # Cantilever
        risk_factors.append("Cantilever support type requires careful monitoring")
    if env_condition in [3, 4]:  # Coastal Area or Industrial Area
        risk_factors.append("Environmental conditions may accelerate structural deterioration")
    
    # Generate technical recommendations
    recommendations = []
    if health_score < 70:
        recommendations.append("Consider upgrading concrete grade for better structural integrity")
        recommendations.append("Implement additional support systems if possible")
        recommendations.append("Schedule regular structural inspections")
    if seismic_load_enabled:
        recommendations.append("Install seismic dampers or base isolators")
    if env_condition in [3, 4]:
        recommendations.append("Apply protective coatings to prevent corrosion")
        recommendations.append("Implement drainage systems to manage water exposure")
    
    # Generate detailed analysis
    analysis = f"""
    <p><strong>Structural Analysis Summary:</strong></p>
    <ul>
        <li>Current structural health score: {health_score:.1f}%</li>
        <li>Building elevation: {elevation} meters</li>
        <li>Applied load: {applied_load} kN</li>
        <li>Concrete grade: {CONCRETE_GRADES[concrete_grade]}</li>
        <li>Support type: {SUPPORT_TYPES[support_type]}</li>
        <li>Environmental condition: {ENV_CONDITIONS[env_condition]}</li>
    </ul>
    <p><strong>Key Observations:</strong></p>
    <ul>
        <li>Seismic considerations: {'Active' if seismic_load_enabled else 'Inactive'}</li>
        <li>Load capacity: {'Adequate' if applied_load <= 500 else 'High'}</li>
        <li>Environmental impact: {'Low' if env_condition == 0 else 'Moderate' if env_condition in [1, 2] else 'High'}</li>
    </ul>
    """
    
    return jsonify({
        'health_score': round(health_score, 1),
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'analysis': analysis
    })

@app.route('/about')
def about():
    return render_template('about.html')

def generate_pdf_report(data, health_score, risk_factors, recommendations, analysis):
    # Create a buffer to store the PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    elements.append(Paragraph("Structural Health Assessment Report", title_style))
    
    # Date
    date_style = ParagraphStyle(
        'Date',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.gray
    )
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
    elements.append(Spacer(1, 20))
    
    # Health Score
    elements.append(Paragraph("Structural Health Score", styles['Heading2']))
    elements.append(Paragraph(f"{health_score:.1f}%", styles['Heading3']))
    elements.append(Spacer(1, 20))
    
    # Condition Assessment
    elements.append(Paragraph("Condition Assessment", styles['Heading2']))
    condition = "GOOD" if health_score >= 70 else "MODERATE" if health_score >= 50 else "BAD"
    safety = "SAFE" if health_score >= 70 else "ACTIONS REQUIRED" if health_score >= 50 else "UNSAFE"
    
    condition_data = [
        ["Condition", condition],
        ["Safety Status", safety]
    ]
    
    condition_table = Table(condition_data, colWidths=[2*inch, 2*inch])
    condition_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, 1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.black),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, 1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(condition_table)
    elements.append(Spacer(1, 20))
    
    # Building Parameters
    elements.append(Paragraph("Building Parameters", styles['Heading2']))
    param_data = [
        ["Parameter", "Value"],
        ["Seismic Load", "ON" if data['seismic_load_enabled'] == 1 else "OFF"],
        ["Seismic Load Factor", str(data['seismic_load'])],
        ["Concrete Grade", data['concrete_grade']],
        ["Elevation", f"{data['elevation']} meters"],
        ["Applied Load", f"{data['applied_load']} kN"],
        ["Support Type", data['support_type']],
        ["Environmental Condition", data['env_condition']]
    ]
    
    param_table = Table(param_data, colWidths=[2*inch, 2*inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(param_table)
    elements.append(Spacer(1, 20))
    
    # Risk Factors
    elements.append(Paragraph("Risk Assessment", styles['Heading2']))
    for risk in risk_factors:
        elements.append(Paragraph(f"• {risk}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Technical Recommendations
    elements.append(Paragraph("Technical Recommendations", styles['Heading2']))
    for rec in recommendations:
        elements.append(Paragraph(f"• {rec}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Detailed Analysis
    elements.append(Paragraph("Detailed Analysis", styles['Heading2']))
    elements.append(Paragraph(analysis, styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

@app.route('/download-report', methods=['POST'])
def download_report():
    data = request.json
    health_score = data['health_score']
    risk_factors = data['risk_factors']
    recommendations = data['recommendations']
    analysis = data['analysis']
    
    # Generate PDF
    pdf_buffer = generate_pdf_report(data, health_score, risk_factors, recommendations, analysis)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'structural_health_report_{timestamp}.pdf'
    
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    app.run(debug=True) 