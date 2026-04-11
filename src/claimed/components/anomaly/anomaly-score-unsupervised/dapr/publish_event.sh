#!/bin/bash
dapr publish --publish-app-id anomalydetection --pubsub pubsub --topic anomaly-data --data '{"orderId": "100"}'