FROM python:3.7
# setup
RUN mkdir /app && mkdir /data
WORKDIR /app
# install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# install app
COPY . .
RUN pip install --no-cache-dir -e ".[test]"
# setup dedicated user
RUN adduser --disabled-password --gecos '' app-user
RUN chown -R app-user:app-user /app /data
USER app-user
# setup UTF-8 locale
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LC_CTYPE=C.UTF-8
# image default command
CMD ["/bin/bash"]
