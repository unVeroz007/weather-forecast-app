#!/bin/bash

# Setup script for Streamlit Cloud deployment
# This script configures the environment for the weather forecasting app

echo "🔧 Setting up Weather Forecasting App environment..."

# Create .streamlit directory if it doesn't exist
mkdir -p ~/.streamlit/

# Create config.toml for Streamlit
echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
enableWebsocketCompression = true\n\
\n\
[browser]\n\
serverAddress = \"0.0.0.0\"\n\
serverPort = \$PORT\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
primaryColor = \"#1f77b4\"\n\
backgroundColor = \"#ffffff\"\n\
secondaryBackgroundColor = \"#f0f2f6\"\n\
textColor = \"#31333f\"\n\
font = \"sans serif\"\n\
" > ~/.streamlit/config.toml

echo "✅ Streamlit configuration created!"

# Check Python version
echo "🐍 Python version:"
python --version

# Check if model files exist, if not generate them
if [ ! -f "model_rf.pkl" ] || [ ! -f "scaler.pkl" ]; then
    echo "🔨 Model files not found. Generating demo model..."
    python train_model.py
    if [ $? -eq 0 ]; then
        echo "✅ Demo model generated successfully!"
    else
        echo "❌ Failed to generate model. Check train_model.py"
    fi
else
    echo "✅ Model files found."
    echo "📊 Model size:"
    ls -lh model_rf.pkl scaler.pkl
fi

echo "🎉 Setup completed! Starting Streamlit app..."
echo "🌐 App will be available at: http://localhost:\$PORT"