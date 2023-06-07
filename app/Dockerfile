# Base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the code file and pickle files to the working directory
COPY app.py /app/
COPY tfidf_remitter_name_m3.pkl /app/
COPY tfidf_source_m3.pkl /app/
COPY tfidf_base_txn_text_m3.pkl /app/
COPY tfidf_mode_m3.pkl /app/
COPY tfidf_benef_name_m3.pkl /app/
COPY classifier_m3.pkl /app/

# Install the required packages
RUN pip install --no-cache-dir scikit-learn pandas numpy Flask

# Expose the port on which the Flask app will run (if needed)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
