FROM registry.access.redhat.com/ubi8/python-39
USER root
RUN dnf -y install java-1.8.0-openjdk maven
USER default
RUN wget https://dlcdn.apache.org/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz
RUN tar xvfz spark-3.3.0-bin-hadoop3.tgz
ENV SPARK_HOME=/opt/app-root/src/spark-3.3.0-bin-hadoop3
ENV PATH=$PATH:$SPARK_HOME/bin
RUN git clone https://github.com/CODAIT/stocator.git
RUN cd /opt/app-root/src/stocator && git checkout v1.1.5
RUN cd /opt/app-root/src/stocator && mvn clean package -Pall-in-one
RUN cd /opt/app-root/src/stocator && mvn dependency:tree
#ADD spark-sql-interactive/stocator-1.1.5-jar-with-dependencies.jar .
RUN pip install pyspark==3.3.0 apache-iceberg==0.0.2 ipython nbformat jupyter nbconvert ipykernel
ADD spark-sql-interactive.ipynb .
RUN jupyter nbconvert --to script spark-sql-interactive.ipynb  
CMD python -m ipykernel_launcher -f $DOCKERNEL_CONNECTION_FILE