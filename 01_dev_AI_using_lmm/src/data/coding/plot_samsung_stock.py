# filename: plot_samsung_stock.py
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 삼성전자 주식 코드 (Yahoo Finance에서는 '005930.KS'로 표시됩니다)
stock_code = '005930.KS'

# 오늘 날짜와 3개월 전 날짜 계산
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

# yfinance를 사용하여 주식 데이터 가져오기
data = yf.download(stock_code, start=start_date, end=end_date)

# 그래프 그리기
fig = go.Figure()

# 종가 데이터를 사용하여 그래프에 추가
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], fill='tozeroy', fillcolor='rgba(0,255,0,0.2)', line_color='green', name='Close'))

# y축은 구간의 최소값에서 시작하도록 설정
fig.update_yaxes(range=[data['Close'].min(), data['Close'].max()])

# 그래프 레이아웃 설정
fig.update_layout(title='Samsung Electronics Stock Price Last 3 Months', xaxis_title='Date', yaxis_title='Stock Price (KRW)', template='plotly_white')

# 그래프를 이미지 파일로 저장
fig.write_image("samsung_stock_price.png")