#!/bin/bash  
docker build -t spark-sql-interactive:`echo $version` -f spark-sql-interactive/Dockerfile .
docker tag spark-sql-interactive:`echo $version` `echo $repository`/spark-sql-interactive:`echo $version`
docker push `echo $repository`/spark-sql-interactive:`echo $version`