FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask gunicorn

# Copy files
COPY test.py .
COPY app.py .
COPY Ocean_Health_Index_2018_global_scores.csv .

# Train the model first
RUN python test.py

# Expose port
EXPOSE 5000

# Run Flask app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]