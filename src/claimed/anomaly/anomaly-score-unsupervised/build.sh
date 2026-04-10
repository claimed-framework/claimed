#!/bin/bash
pip install jupyter nbconvert
jupyter nbconvert --to script anomaly-score-unsupervised.ipynb    
docker build -t claimed-anomaly-score-unsupervised:`echo $version` -f anomaly-score-unsupervised/Dockerfile .
docker tag claimed-anomaly-score-unsupervised:`echo $version` `echo $repository`/claimed-anomaly-score-unsupervised:`echo $version`
docker push `echo $repository`/claimed-anomaly-score-unsupervised:`echo $version`
