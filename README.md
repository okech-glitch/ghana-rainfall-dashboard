# 🌾 Agri-Adapt AI: AI-Powered Agricultural Resilience Platform

&nbsp;
Empowering Kenyan farmers with AI-driven drought resilience insights to make informed crop decisions and reduce agricultural losses by up to 30%.

## 🎯 Project Overview

Agri-Adapt AI addresses Kenya's critical food security challenge by providing smallholder farmers with AI-powered maize drought resilience scores. Using historical climate and soil data, our Random Forest model predicts crop resilience, helping farmers make informed planting decisions and reduce crop failures by 20-30%.

## 🌍 Problem Statement

- **Drought Impact**: 30% increase in drought frequency affecting rain-fed agriculture
- **Crop Failures**: 20-30% annual losses in vulnerable counties like Nakuru and Machakos
- **Data Gap**: Siloed climate, soil, and yield data leaves farmers without actionable insights
- **Food Security**: Maize is Kenya's staple crop, critical for national food security

## 🚀 Solution

- **AI-Powered Scoring**: Machine learning model predicts maize resilience (0-100%)
- **Actionable Insights**: Visual gauge with planting recommendations
- **Farmer-Focused**: Simple, mobile-friendly interface for low-literacy users
- **Data-Driven**: Integrates CHIRPS rainfall, AfSIS soil, and FAOSTAT yield data

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   ML Model      │
│   (Next.js/React)│◄──►│   (Python)      │◄──►│   (Random Forest)│
│                 │    │                 │    │                 │
│ • County Select │    │ • /api/predict  │    │ • Rainfall      │
│ • Input Forms   │    │ • /api/counties │    │ • Soil pH       │
│ • Gauge Chart   │    │ • Validation    │    │ • Organic Carbon│
│ • Results       │    │ • Error Handling│    │ • Yield Output  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🌟 Current Features

### 🎯 Core Functionality
- **Drought Resilience Scoring**: AI-powered county-level resilience scores (0-100%)
- **Interactive County Selection**: Easy-to-use dropdown with 20 Kenyan counties
- **Real-Time Predictions**: Sub-second response times for score calculations
- **Mobile-First Design**: Optimized for smartphones with responsive UI

### 📊 Data Visualization
- **Resilience Gauge**: Visual representation of drought resilience score
- **Weather Integration**: Monthly weather data for selected counties
- **Interactive Charts**: Weather patterns and yield predictions
- **County Comparison**: Side-by-side analysis of different regions

### 🗺️ Geographic Coverage
- **20 Kenyan Counties**: Complete coverage of major agricultural regions
- **County-Specific Data**: Tailored insights based on local conditions
- **Regional Analysis**: Comparative resilience across different zones

### 🔧 Technical Features
- **FastAPI Backend**: High-performance Python API with automatic documentation
- **Next.js Frontend**: Modern React framework with TypeScript
- **Machine Learning**: Random Forest model with 70% accuracy
- **Data Integration**: CHIRPS, AfSIS, and FAOSTAT data sources

## 🚀 Proposed Features (Roadmap)

### 📱 Enhanced User Experience
- **Multi-Language Support**: Swahili and English interfaces
- **Offline Capability**: PWA with cached data for remote areas
- **Voice Input**: Speech-to-text for illiterate users
- **SMS Integration**: Text-based access for basic phones

### 📊 Advanced Analytics
- **Historical Trends**: Multi-year resilience tracking
- **Climate Forecasting**: Integration with weather prediction models
- **Crop Diversification**: Recommendations for alternative crops
- **Risk Assessment**: Probability of crop failure scenarios

### 🤖 AI Enhancements
- **Deep Learning Models**: Neural networks for improved accuracy
- **Satellite Imagery**: Remote sensing for real-time crop monitoring
- **Predictive Maintenance**: Equipment and irrigation system optimization
- **Natural Language Queries**: Conversational AI for farmer questions

### 🌍 Expanded Coverage
- **Regional Expansion**: Coverage beyond Kenya to East Africa
- **Crop Variety**: Support for multiple crops (beans, sorghum, etc.)
- **Soil Health Monitoring**: Continuous soil quality assessment
- **Market Integration**: Price prediction and market access

## 🧮 How the Resilience Score is Calculated

The resilience score is calculated using a machine learning model that predicts maize yield and converts it to a percentage score. Here's the detailed process:

### 1. Input Features (14 Numerical Features + County Encoding)

**User Inputs:**
- **Rainfall**: Annual rainfall in mm (e.g., 800mm)
- **Soil pH**: Soil acidity/alkalinity (e.g., 6.5)
- **Organic Carbon**: Soil organic matter % (e.g., 2.1%)

**County-Specific Data (Automatically Loaded):**
- **Temperature**: Monthly average and standard deviation (°C)
- **Humidity**: Monthly average and standard deviation (%)
- **Precipitation**: Monthly average and standard deviation (mm)
- **Soil Properties**: Clay content, silt, sand percentages
- **Climate Variability**: Temperature and precipitation variation coefficients

### 2. Machine Learning Model

- **Algorithm**: Random Forest Regressor
- **Training Data**: Historical climate, soil, and yield data from 2019-2023
- **Cross-Validation**: 5-fold CV with consistent performance
- **Accuracy**: R² Score of 0.7 (70% accuracy)

### 3. Score Calculation Formula

```
Resilience Score (%) = (Predicted Yield / Benchmark Yield) × 100
```

Where:
- **Predicted Yield** = ML model output (tons/hectare)
- **Benchmark Yield** = 2.5 tons/hectare (Kenya's average maize yield)

### 4. Score Interpretation

| Score Range | Resilience Level | Recommendation |
|-------------|------------------|----------------|
| 80-100% | High Resilience | Optimal conditions, proceed with planting |
| 60-79% | Moderate Resilience | Good conditions, consider drought-resistant varieties |
| 40-59% | Low Resilience | Challenging conditions, implement water conservation |
| 0-39% | Very Low Resilience | High risk, consider alternative crops or delay |

### 5. Feature Importance

The model prioritizes these factors in order:
- **Rainfall** (35% importance) - Most critical for drought resilience
- **Soil pH** (25% importance) - Affects nutrient availability
- **Temperature Variability** (20% importance) - Climate stress indicator
- **Organic Carbon** (15% importance) - Soil health and water retention
- **County-Specific Factors** (5% importance) - Local agricultural conditions

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- Git

### Installation

#### Clone the repository
```bash
git clone https://github.com/okech-glitch/ghana-rainfall-dashboard.git
cd ghana-rainfall-dashboard
```

#### Backend Setup (Python/FastAPI)
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

#### Frontend Setup (Next.js/React)
```bash
cd frontend
# Install Node.js dependencies
npm install
```

#### Start the Backend
```bash
cd ..
python app.py
# Backend will run on http://localhost:5000
```

#### Start the Frontend
```bash
cd frontend
npm run dev
# Frontend will run on http://localhost:3000
```

## 🏗️ Project Structure

```
agri-adapt-ai/
├── 📁 frontend/                 # Next.js frontend application
│   ├── app/                     # Next.js app directory
│   │   ├── page.tsx            # Main dashboard page
│   │   ├── layout.tsx          # Root layout
│   │   └── globals.css         # Global styles
│   ├── components/              # React components
│   │   ├── ui/                 # Reusable UI components
│   │   ├── resilience-gauge.tsx # Resilience score display
│   │   ├── recommendations-panel.tsx # Farming recommendations
│   │   ├── data-visualization.tsx    # Charts and graphs
│   │   └── weather-integration.tsx   # Weather data integration
│   ├── lib/                    # Utility functions
│   └── public/                 # Static assets
├── 📁 src/                      # Python backend source
│   ├── api/                    # FastAPI application
│   │   ├── fastapi_app.py     # Main FastAPI app
│   │   ├── data_service.py    # Data processing service
│   │   └── weather_service.py # Weather data service
│   ├── models/                 # ML model classes
│   │   └── maize_resilience_model.py # Main ML model
│   └── utils/                  # Utility functions
├── 📁 config/                   # Configuration files
│   └── settings.py             # Application settings
├── 📁 scripts/                  # Training and utility scripts
│   ├── analysis/               # Data analysis scripts
│   ├── data_processing/        # Data processing scripts
│   ├── modeling/               # Model training scripts
│   └── utilities/              # Utility scripts
├── 📁 tests/                    # Test suites
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── 📁 docs/                     # Documentation
│   ├── api/                    # API documentation
│   ├── technical/              # Technical documentation
│   └── user_guide/             # User guides
├── 📁 requirements.txt          # Python dependencies
└── 📁 README.md                 # This file
```

## 🔧 Backend API Endpoints

### Core Endpoints
- **GET /health** - System health check
- **GET /api/counties** - List of Kenya counties
- **POST /api/predict** - Single prediction
- **POST /api/predict/batch** - Batch predictions
- **GET /api/model/status** - Model performance info
- **GET /api/metrics** - Usage statistics

### Weather Endpoints
- **GET /api/weather/{county}/monthly** - Monthly weather data
- **GET /api/weather/{county}/current** - Current weather conditions

### Example API Usage

```bash
# Make a prediction
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "rainfall": 800,
    "soil_ph": 6.5,
    "organic_carbon": 2.1,
    "county": "Nakuru"
  }'
```

**Response:**
```json
{
  "resilience_score": 75.2,
  "confidence": 0.85,
  "recommendations": [
    "Consider drought-resistant maize varieties",
    "Implement water conservation practices",
    "Monitor soil moisture regularly"
  ]
}
```

## 🎨 Frontend Features

### Dashboard Components
- **Resilience Gauge**: Visual representation of drought resilience score
- **County Selector**: Interactive dropdown with search functionality
- **Recommendations Panel**: Actionable farming advice based on scores
- **Data Visualization**: Interactive charts for weather and yield data
- **Weather Integration**: Real-time weather data for selected counties
- **Cost Calculator**: Input cost analysis for different farming strategies

### Technology Stack
- **Framework**: Next.js 15 with App Router
- **UI Library**: React 18 with TypeScript
- **Styling**: Tailwind CSS 4 with custom components
- **Components**: Radix UI for accessibility
- **Charts**: Recharts for data visualization
- **Forms**: React Hook Form with Zod validation

## 🧪 Testing

### Backend Testing
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/unit/test_backend_api.py

# Run with coverage
python -m pytest --cov=src
```

### Frontend Testing
```bash
cd frontend
# Run tests
npm test

# Run with coverage
npm run test:coverage
```

## 🚀 Deployment

### Backend Deployment
```bash
# Using Docker
docker build -t agri-adapt-ai-backend .
docker run -p 8000:8000 agri-adapt-ai-backend

# Using Docker Compose
docker-compose up -d
```

### Frontend Deployment
```bash
cd frontend
# Build for production
npm run build

# Start production server
npm start

# Deploy to Vercel
vercel --prod
```

## 📊 Model Performance

- **R² Score**: 0.7 (70% accuracy)
- **Algorithm**: Random Forest Regressor
- **Features**: 14 numerical features + county encoding
- **Training Data**: Historical climate and soil data (2019-2023)
- **Cross-validation**: 5-fold CV with consistent performance
- **Response Time**: <1 second for predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for frontend components
- Write tests for new features
- Update documentation for API changes
- Use conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kenyan farmers for their valuable insights
- Climate data providers (CHIRPS, AfSIS, FAOSTAT)
- Open-source community for tools and libraries
- Agricultural experts for domain knowledge

## 📞 Support

- **Documentation**: API Docs
- **Issues**: GitHub Issues
- **Email**: support@agri-adapt-ai.com

Built with ❤️ for sustainable agriculture in Kenya

## 🔄 Recent Updates

- **v1.2.0**: Added county-specific weather data integration
- **v1.1.0**: Enhanced ML model with improved accuracy
- **v1.0.0**: Initial release with core resilience scoring

---

**About**
No description, website, or topics provided.

**Resources**
- 📖 Readme
- 📋 License
- 🤝 Contributing
- 🔒 Security policy

**Activity**
- ⭐ 0 stars
- 👀 0 watching
- 🍴 1 fork

**Report repository**

**Releases**
No releases published

**Packages**
No packages published

**Languages**
- Python 65.9%
- TypeScript 28.0%
- JavaScript 2.7%
- CSS 2.6%
- Other 0.8%
