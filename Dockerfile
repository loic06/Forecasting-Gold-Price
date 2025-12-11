FROM python:3.10.6-buster
COPY requirements.txt /requirements.txt
COPY api /api
COPY forecasting_gold_price /forecasting_gold_price
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# RUN mkdir /raw_data
# RUN mkdir /models 'si utilisation d'un model'
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
