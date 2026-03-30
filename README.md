# 🌾 Crop Recommendation System

A machine learning-powered web application that recommends suitable crops based on soil and environmental conditions. The system uses multiple ML models to provide accurate predictions and detailed crop information.

## 🚀 Features

### Multiple ML Models
- **Random Forest** (Default) - Best all-around model with feature importance analysis
- **Support Vector Machine** - Good for complex decision boundaries
- **Logistic Regression** - Simple and interpretable
- **K-Nearest Neighbors** - Pattern-based prediction

### Detailed Analysis
- Crop recommendation with confidence levels
- Feature importance visualization (Random Forest)
- Comparison of input conditions with optimal conditions
- Comprehensive crop information including:
  - Growing season
  - Water requirements
  - Soil type recommendations
  - Fertilizer needs
  - Optimal growing conditions

### User Interface
- Clean, responsive Bootstrap design
- Real-time model selection
- Interactive visualization of predictions
- Loading states and error handling
- Mobile-friendly layout

## 🛠️ Technical Stack

### Backend
- FastAPI
- scikit-learn
- NumPy
- joblib

### Frontend
- HTML5
- CSS3 (Bootstrap 5)
- JavaScript (Vanilla)

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Web browser (Chrome/Firefox/Safari)

## ⚙️ Installation




3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

1. Train the ML models:
   ```bash
   python backend/train_models.py
   ```
   This will create trained models in the `backend/models` directory.

2. Start the backend server:
   ```bash
   # From the project root
   uvicorn backend.main:app --reload
   ```
   The API will be available at `http://localhost:8000`

3. Open the frontend:
   - Navigate to the `frontend` directory
   - Open `index.html` in your web browser
   - Or use a local server:
     ```bash
     # Using Python's built-in server
     cd frontend
     python -m http.server 8080
     ```
   Then visit `http://localhost:8080`

## 🎯 Using the Application

1. Select a Machine Learning Model:
   - Random Forest (recommended for most cases)
   - SVM (for complex patterns)
   - Logistic Regression (for simple patterns)
   - KNN (for pattern-based prediction)

2. Enter Environmental Data:
   - Soil pH (0-14)
   - Temperature (°C)
   - Humidity (%)
   - Rainfall (mm)

3. View Results:
   - Recommended crop with confidence level
   - Comparison with optimal conditions
   - Feature importance (Random Forest only)
   - Detailed crop information

## 🌱 Supported Crops

The system can recommend the following crops:
- Rice
- Wheat
- Corn
- Sugarcane

Each crop recommendation includes:
- Growing season duration
- Water requirements
- Soil type preferences
- Fertilizer recommendations
- Optimal environmental conditions

## 📊 Model Accuracy

The models are trained on a curated dataset with the following characteristics:
- Multiple examples per crop
- Standardized features
- Balanced class distribution
- Cross-validated accuracy

## 🔄 Updating the System

### Adding New Crops
1. Update `backend/train_models.py`:
   - Add new crop data to training set
   - Retrain models

2. Update `backend/main.py`:
   - Add crop details to `CROP_DETAILS` dictionary

### Modifying Models
1. Edit `backend/train_models.py`:
   - Adjust model parameters
   - Add/modify features
   - Change training data

2. Retrain models:
   ```bash
   python backend/train_models.py
   ```

### Frontend Customization
- Edit `frontend/styles.css` for styling
- Modify `frontend/app.js` for behavior
- Update `frontend/index.html` for layout

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Troubleshooting

### Common Issues

1. Models not found:
   ```bash
   python backend/train_models.py
   ```

2. CORS errors:
   - Check that backend is running on correct port
   - Verify CORS settings in `backend/main.py`

3. Prediction errors:
   - Verify input data is within valid ranges
   - Check model files exist in `backend/models`

### Getting Help

- Open an issue for bugs
- Check existing issues for solutions
- Contact maintainers for support 


The CSV file contains training data with features (N, P, K, temperature, humidity, pH, rainfall) and the corresponding crop label
This data is used to train multiple machine learning models (Random Forest, SVM, Logistic Regression, and KNN)
The trained models are saved as .joblib files in the models/ directory
Making Predictions (main.py):
When you make a request to predict a crop, the system:
Takes your input values (temperature, humidity, pH, rainfall)
Uses the trained models (loaded from the .joblib files) to predict the best crop
Returns the prediction along with growing conditions and other details
So while the CSV file is crucial for training the models, it's not directly used for making recommendations. Instead, the trained models (which learned patterns from this data) are used to make the predictions.
Looking at the CSV data, we can see it contains real agricultural data with:
Soil nutrients (N, P, K)
Environmental conditions (temperature, humidity, pH, rainfall)
Target crop labels (rice, maize, etc.)