#!/bin/bash
dapr run --app-id anomalydetection --app-port 6002 --dapr-http-port 3602 --app-protocol grpc -- python3 subscriber.py