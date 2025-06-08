# DIABETES RISK ANALYZER

An AI-powered intelligent healthcare solution that provides personalized diabetes risk assessment and preventive measures through an interactive conversational interface. The system leverages machine learning algorithms to analyze health metrics and deliver customized health recommendations.

## 🌟 Key Features

### 1. Intelligent Conversational Interface
- **Gender-Aware Questioning System**
  - Smart gender-based question flow
  - Contextual pregnancy questions for females
  - Automated question adaptation
  - Natural conversation flow

- **Dynamic Health Assessment**
  - Real-time input validation
  - Context-aware follow-up questions
  - Medical range validation
  - Interactive feedback system

### 2. Comprehensive Health Analysis
- **Core Health Metrics**
  - Gender and age assessment
  - Pregnancy history (for females)
  - Glucose level monitoring
  - Blood pressure tracking
  - Body composition analysis
  - Insulin level assessment
  - BMI calculation
  - Family history evaluation

- **Advanced Analytics**
  - Machine learning-based risk prediction
  - Personalized risk assessment
  - Confidence scoring
  - Trend analysis

### 3. Smart Validation System
- **Real-time Validation**
  - Instant input verification
  - Medical range compliance
  - Clear error messaging
  - Format flexibility

- **Medical Context**
  - Standard medical ranges
  - Professional guidelines
  - Best practice recommendations
  - Safety thresholds

### 4. Intelligent Follow-up System
- **Contextual Questions**
  - Glucose level follow-ups
  - Blood pressure monitoring
  - BMI-based inquiries
  - Age-specific considerations

- **Detailed Assessment**
  - Medical history collection
  - Symptom tracking
  - Lifestyle evaluation
  - Family history analysis
  - Medication review
  - Diet and exercise assessment

### 5. Personalized Prevention
- **Customized Recommendations**
  - Risk-based guidelines
  - Profile-specific advice
  - Age-appropriate measures
  - Lifestyle modifications

- **Comprehensive Care Plan**
  - General health guidelines
  - High-risk protocols
  - Age-specific recommendations
  - Lifestyle adjustments
  - Emergency guidelines
  - Medical management plans

### 6. Voice Interaction
- **Advanced Voice Processing**
  - Speech-to-text conversion
  - Multi-accent support
  - Real-time processing
  - Voice command recognition

## 🛠 Technical Architecture

### Backend Infrastructure
- **Core Technology**: Flask (Python)
- **Machine Learning**: Random Forest Classifier
- **API Design**: RESTful architecture
- **State Management**: Session-based
- **Data Processing**: Real-time validation
- **Error Handling**: Comprehensive system

### API Endpoints
```
POST /api/process-text     # Text input processing
POST /api/process-voice    # Voice input handling
GET  /api/predict         # Risk prediction
GET  /api/preventive      # Preventive measures
POST /api/reset          # Session management
GET  /api/current-state  # Question state
```

### Input Validation
- **Numeric Validation**
  - Medical measurement ranges
  - Age restrictions
  - BMI calculations
  - Blood pressure norms

- **Categorical Validation**
  - Gender options
  - Yes/No responses
  - Medical history
  - Lifestyle factors

### Response Structure
- Success/Error status
- Current question state
- Follow-up handling
- User feedback
- Prediction results
- Preventive measures

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Flask framework
- Required Python packages (see requirements.txt)
- Modern web browser
- JavaScript enabled

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-risk-analyzer.git
   cd diabetes-risk-analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   - Create `.env` file
   - Add configuration (see .env.example)

### Running the Application
1. Start the backend server:
   ```bash
   python backend/app.py
   ```

2. Access the frontend:
   - Open web browser
   - Navigate to localhost:5000

## 📝 Usage Guide

1. **Initial Setup**
   - Launch the application
   - Select gender
   - Begin health assessment

2. **Health Assessment**
   - Answer health questions
   - Provide medical data
   - Respond to follow-ups
   - Review recommendations

3. **Results and Recommendations**
   - View risk assessment
   - Review preventive measures
   - Access detailed guidelines
   - Save recommendations

## ⚠️ Medical Disclaimer

This application is designed for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Documentation

For detailed documentation, please refer to:
- [Project Report](PROJECT_REPORT.md)
- [API Documentation](docs/API.md)
- [User Guide](docs/USER_GUIDE.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.md)

## 📞 Support

For support, please:
- Open an issue
- Contact the development team
- Check the FAQ section
- Review documentation

## 🙏 Acknowledgments

- Pima Indians Diabetes Database
- Open-source community
- Healthcare professionals
- Beta testers
- Contributors 