// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // Constants for API endpoints
    const API_BASE_URL = 'http://localhost:8000';
    const PREDICT_ENDPOINT = `${API_BASE_URL}/predict`;

    const form = document.getElementById('cropForm');
    const predictionDiv = document.getElementById('prediction');
    const initialMessage = document.getElementById('initial-message');
    const modelSelect = document.getElementById('modelSelect');
    const modelDescription = document.getElementById('modelDescription');

    // Model descriptions
    const MODEL_DESCRIPTIONS = {
        'rf': 'Random Forest - Good all-around model with feature importance',
        'svm': 'Support Vector Machine - Good for complex decision boundaries',
        'lr': 'Logistic Regression - Simple and interpretable',
        'knn': 'K-Nearest Neighbors - Good for pattern-based prediction'
    };

    // Update model description when selection changes
    modelSelect.addEventListener('change', (e) => {
        modelDescription.textContent = MODEL_DESCRIPTIONS[e.target.value];
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        initialMessage.style.display = 'none';
        predictionDiv.style.display = 'none';
        predictionDiv.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-2">Analyzing your input...</p>
            </div>
        `;
        predictionDiv.style.display = 'block';
        
        const formData = {
            N: 0, // Default value since N is not required
            P: 0, // Default value since P is not required
            K: 0, // Default value since K is not required
            temperature: parseFloat(document.getElementById('temperature').value),
            humidity: parseFloat(document.getElementById('humidity').value),
            ph: parseFloat(document.getElementById('ph').value),
            rainfall: parseFloat(document.getElementById('rainfall').value)
        };

        try {
            const selectedModel = modelSelect.value;
            const response = await fetch(`${PREDICT_ENDPOINT}?model_name=${selectedModel}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                mode: 'cors',
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const data = await response.json();
            
            // Update prediction display
            let html = `
                <div class="alert alert-success mb-4">
                    <h4 class="alert-heading">Recommended Crop: ${data.predicted_crop}</h4>
                    <p class="mb-0"><small>Model used: ${data.model_used}</small></p>
                </div>

                <div class="mb-4">
                    <h5>Input Analysis</h5>
                    <div class="card">
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Your Conditions:</h6>
                                    <ul class="list-unstyled">
                                        ${Object.entries(data.input_analysis.provided_conditions)
                                            .filter(([key]) => !['N', 'P', 'K'].includes(key))
                                            .map(([key, value]) => `
                                                <li><strong>${key}:</strong> ${value}</li>
                                            `).join('')}
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <h6>Optimal Conditions:</h6>
                                    <p>${data.input_analysis.optimal_conditions}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mb-4">
                    <h5>Prediction Confidence:</h5>
                    <ul class="list-group">
                        ${Object.entries(data.probabilities)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 5)
                            .map(([crop, probability]) => `
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    ${crop}
                                    <div class="ms-2">
                                        <div class="progress" style="width: 100px;">
                                            <div class="progress-bar ${probability > 50 ? 'bg-success' : ''}" 
                                                role="progressbar" 
                                                style="width: ${probability}%" 
                                                aria-valuenow="${probability}" 
                                                aria-valuemin="0" 
                                                aria-valuemax="100">
                                                ${probability}%
                                            </div>
                                        </div>
                                    </div>
                                </li>
                            `).join('')}
                    </ul>
                </div>

                <div class="crop-details">
                    <h5>Crop Information:</h5>
                    <div class="card">
                        <div class="card-body">
                            ${Object.entries(data.crop_details)
                                .map(([key, value]) => {
                                    if (key === 'optimal_conditions' || key === 'nutrient_needs') return '';
                                    const formattedKey = key
                                        .split('_')
                                        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                                        .join(' ');
                                    return `<p><strong>${formattedKey}:</strong> ${value}</p>`;
                                })
                                .filter(item => item !== '')
                                .join('')}
                        </div>
                    </div>
                </div>
            `;

            predictionDiv.innerHTML = html;
            initialMessage.style.display = 'none';
            predictionDiv.style.display = 'block';

        } catch (error) {
            console.error('Error:', error);
            predictionDiv.innerHTML = `
                <div class="alert alert-danger">
                    <h4 class="alert-heading">Connection Error</h4>
                    <p>Unable to connect to the prediction server. Please ensure:</p>
                    <ul>
                        <li>The backend server is running at ${API_BASE_URL}</li>
                        <li>You're accessing the frontend via http://localhost:8080/frontend/index.html</li>
                        <li>No other services are using ports 8000 or 8080</li>
                    </ul>
                    <hr>
                    <p class="mb-0">Technical details: ${error.message}</p>
                </div>
            `;
            initialMessage.style.display = 'none';
            predictionDiv.style.display = 'block';
        }
    });
}); 