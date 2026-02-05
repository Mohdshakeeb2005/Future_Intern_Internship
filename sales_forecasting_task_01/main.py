from prophet import Prophet
import pandas as pd

df = pd.read_csv("dataset.csv", encoding="latin1")

model_df = df[['Order Date', 'Sales']].copy()
model_df.columns = ['ds','y']

model_df['ds'] = pd.to_datetime(model_df['ds'], format='mixed', dayfirst=False)

model = Prophet()
model.fit(model_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

forecast[['ds','yhat']].to_csv("forecast.csv", index=False)
