# Diabetes Risk Predictor

A comprehensive web application that predicts diabetes risk using machine learning and provides personalized health recommendations. The system uses a conversational interface to gather health information and provides detailed preventive measures based on the user's profile.

## Features

### 1. Intelligent Question Flow
- Gender-aware questioning system
  - First asks for gender (male/female)
  - Automatically handles pregnancy questions based on gender
  - Skips pregnancy question for males (sets to 0)
  - Asks pregnancy question only for females
- Context-aware follow-up questions
  - Triggers based on specific health measurements
  - Gathers additional context for personalized recommendations
  - Validates responses with appropriate ranges

### 2. Health Metrics Collection
- Comprehensive health data collection:
  - Gender
  - Pregnancy history (for females)
  - Glucose levels
  - Blood pressure
  - Skin thickness
  - Insulin levels
  - BMI
  - Diabetes pedigree function
  - Age

### 3. Smart Validation and Feedback
- Real-time validation of inputs
- Immediate feedback for out-of-range values
- Medical context for measurements
- Clear error messages with valid ranges
- Support for various input formats

### 4. Follow-up Questions System
- Context-aware follow-up questions for:
  - High/Low glucose levels
  - High/Low blood pressure
  - High/Low BMI
  - Age-specific considerations
- Detailed follow-up for:
  - Medical history
  - Symptoms
  - Lifestyle factors
  - Family history
  - Current medications
  - Diet and exercise habits

### 5. Preventive Measures
- Personalized recommendations based on:
  - Risk assessment
  - Health profile
  - Age group
  - Lifestyle factors
- Comprehensive preventive measures including:
  - General health guidelines
  - High-risk individual recommendations
  - Age-specific considerations
  - Lifestyle modifications
  - Emergency preparedness
  - Medical management guidelines

### 6. Voice Input Support
- Speech-to-text conversion
- Support for voice responses
- Handles various accents and speech patterns
- Real-time voice processing

## Technical Details

### Backend
- Built with Flask (Python)
- Machine learning model for risk prediction
- RESTful API endpoints
- Session management for conversation state
- Input validation and processing
- Comprehensive error handling

### API Endpoints
- `/api/process-text`: Process text input
- `/api/process-voice`: Process voice input
- `/api/predict`: Get diabetes risk prediction
- `/api/preventive-measures`: Get personalized preventive measures
- `/api/reset`: Reset the conversation
- `/api/current-question`: Get current question

### Input Validation
- Numeric ranges for medical measurements
- Categorical inputs for lifestyle questions
- Yes/No question handling
- Gender input processing
- Custom validation for each question type

### Response Types
- Success/Error status
- Current question information
- Follow-up question handling
- Feedback messages
- Prediction results
- Preventive measures

## Getting Started

### Prerequisites
- Python 3.8+
- Flask
- Required Python packages (see requirements.txt)
- Web browser with JavaScript enabled

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file
   - Add necessary configuration (see .env.example)

### Running the Application
1. Start the backend server:
   ```bash
   python backend/app.py
   ```
2. Open the frontend in a web browser
3. Start the conversation by providing your gender

## Usage

1. Start the conversation by entering your gender
2. Answer health-related questions
3. Provide additional information when asked
4. Receive personalized risk assessment
5. Get detailed preventive measures
6. Request more specific information about any area

## Medical Disclaimer

This application is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 