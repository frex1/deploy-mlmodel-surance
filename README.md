# deploy-mlmodel-surance

A deployed machine learning web application for insurance cost prediction using pycaret and Flask.

---

## Features

- **Insurance Cost Prediction**: Predicts insurance costs based on user input using a trained ML model.
- **Web Interface**: User-friendly input form via Flask/Jinja2 (see `index.html`).
- **Pre-trained Model**: Loads a production-ready model (`insurance_deploy.pkl`) for fast scoring.
- **Interactive Notebook**: See `webapp.ipynb` for end-to-end dev workflow.
- **Heroku-Ready**: Includes `Procfile` and `gunicorn` config for cloud deployment.

---

## Requirements

- Python 3.7+
- Flask 1.1.1, pycaret 1.0.0
- gunicorn (for production)
- See `requirements.txt` for full dependencies.

---

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/frex1/deploy-mlmodel-surance.git
   cd deploy-mlmodel-surance
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web app locally**
   ```bash
   python app.py
   ```
   The app will be available at `http://127.0.0.1:5000/`

4. **Deploy on Heroku (optional)**
   - Create your Heroku app and push your code.
   - The included `Procfile` and gunicorn requirement are ready for deployment.

---

## Project Structure

```
.
├── app.py                # Flask application main file
├── insurance_deploy.pkl  # Serialized ML model
├── insurance.csv         # Example training data
├── index.html            # Web form template
├── requirements.txt
├── Procfile              # Heroku deployment script
├── webapp.ipynb          # Dev notebook/workflow
└── ...
```

---

## Usage

- Navigate to the app in your browser.
- Enter the required information and receive predicted insurance costs on form submission.

---

## Author

- [@frex1](https://github.com/frex1)

---

## Issues

Please file issues or feature requests at: https://github.com/frex1/deploy-mlmodel-surance/issues

---

## License

MIT
