FROM registry.access.redhat.com/ubi8/python-39
RUN pip install ipython nbformat tensorflow==2.9.1 numpy==1.23.2 scikit-learn==1.1.2 pandas==1.4.3 flask==2.2.2 matplotlib==3.5.3
ADD anomaly-score-unsupervised.py .
ENTRYPOINT ["python", "anomaly-score-unsupervised.py"]