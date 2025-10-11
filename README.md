# 🌧️ Rainfall Prediction Dashboard

A production-ready rainfall prediction web application using LightGBM ensemble models, perfect for portfolio demonstration and real-world deployment.

## 🚀 Features

- **Advanced ML Models**: 10-fold LightGBM ensemble with meta-learning
- **Real-time Predictions**: Interactive weather parameter input
- **Beautiful UI**: Modern, responsive design
- **Production Ready**: Deployed on Vercel with Flask API
- **High Accuracy**: 89.57% Macro F1 score

## 📊 Model Performance

| Model | Macro F1 | Status |
|-------|----------|--------|
| **LightGBM Ensemble** | **0.895653** | ✅ Production |
| XGBoost | 0.891497 | ✅ Available |
| Random Forest | 0.740136 | ✅ Available |
| TabNet | 0.685452 | ✅ Available |
| CatBoost | 0.658683 | ⚠️ Low Performance |

## 🛠️ Local Development

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Locally
```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## 🌐 Deploy to Vercel

### Method 1: Vercel CLI (Recommended)
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod

# Set environment variables if needed
vercel env add FLASK_ENV
```

### Method 2: GitHub Integration
1. Push code to GitHub repository
2. Connect repository to Vercel
3. Auto-deploy on every push

### Method 3: Manual Deploy
1. Go to [vercel.com](https://vercel.com)
2. Import project from GitHub
3. Deploy with one click

## 📁 Project Structure

```
├── app.py              # Flask API server
├── index.html          # Frontend dashboard
├── requirements.txt    # Python dependencies
├── vercel.json         # Vercel deployment config
├── .vercelignore       # Deployment exclusions
└── models/             # ML models and metadata
    ├── lightgbm_fold*.pkl    # LightGBM models
    ├── label_encoder.npy     # Class labels
    ├── metadata.json         # Model configuration
    └── stacked_meta_model.pkl # Ensemble meta-model
```

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard frontend |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model information |
| `/health` | GET | Health check |

## 📝 Usage Example

### Single Prediction
```javascript
const response = await fetch('/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    temperature: 25.0,
    humidity: 65.0,
    wind_speed: 12.0,
    pressure: 1013.0,
    // ... other parameters
  })
});

const result = await response.json();
// Returns: prediction, confidence, probabilities
```

## 🎨 Customization

### Add New Weather Parameters
Edit the `feature_columns` list in `app.py` and update the HTML form in `index.html`.

### Modify UI Styling
Update the CSS in `index.html` or add custom styles.

### Add New Models
1. Train new models and save as `.pkl` files
2. Update `load_models()` method in `app.py`
3. Add model selection UI in frontend

## 📱 Responsive Design

- ✅ Mobile-friendly interface
- ✅ Tablet optimized
- ✅ Desktop enhanced
- ✅ Touch-friendly controls

## 🔒 Production Considerations

- **Security**: Input validation implemented
- **Performance**: Optimized model loading and inference
- **Scalability**: Stateless design for horizontal scaling
- **Monitoring**: Health check endpoint available
- **Error Handling**: Comprehensive error management

## 📈 Portfolio Value

This project demonstrates:
- **Full-Stack Development**: Python API + HTML/CSS/JavaScript
- **Machine Learning**: Advanced ensemble modeling
- **DevOps**: Production deployment on Vercel
- **UI/UX Design**: Modern, responsive interface
- **Real-world Application**: Weather prediction for agriculture

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - feel free to use for your portfolio!

---

**🌟 Perfect for showcasing ML engineering skills in your portfolio!**
