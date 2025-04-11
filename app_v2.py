import yfinance as yf
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from sklearn.ensemble import IsolationForest
import plotly.graph_objs as go
from datetime import datetime
import database

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Liquidity Dashboard v2"

stock_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# 快取統一更新時間
last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")

# 資料擷取邏輯（含 SQLite 快取）
def fetch_data(ticker):
    df_cached = database.load_from_db(ticker)
    if df_cached is not None:
        return df_cached

    df = yf.download(ticker, period="6mo", interval="1d")
    df.dropna(inplace=True)
    df['Return'] = df['Adj Close'].pct_change()
    df['Amihud'] = (abs(df['Return']) / df['Volume']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Z_Score'] = (df['Amihud'] - df['Amihud'].rolling(20).mean()).fillna(0) / df['Amihud'].rolling(20).std().replace(0, np.nan).fillna(1)
    model = IsolationForest(contamination=0.05, random_state=42)
    df['IF_Anomaly'] = model.fit_predict(df[['Amihud']].fillna(0))
    df = df.reset_index()
    df.rename(columns={'Date': 'date', 'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)
    database.write_to_db(ticker, df)
    df.set_index('date', inplace=True)
    return df

# KPI 卡片元件
def kpi_cards(df):
    latest = df.iloc[-1]
    return dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Amihud Ratio"),
            dbc.CardBody(html.H4(f"{latest['amihud']:.6f}"))
        ]), width=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Volume"),
            dbc.CardBody(html.H4(f"{int(latest['volume']):,}"))
        ]), width=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Anomaly"),
            dbc.CardBody(html.H4("🚨 Yes" if latest['if_anomaly'] == -1 else "✅ No"))
        ]), width=4)
    ])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("📈 Stock Liquidity Dashboard v2"), width=9),
        dbc.Col(dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': s, 'value': s} for s in stock_list],
            value='AAPL', clearable=False
        ), width=3)
    ]),
    html.Div(f"Last Updated: {last_updated}", style={"color": "gray", "fontSize": "0.9rem", "marginBottom": "10px"}),
    dcc.Tabs(id='tabs', value='overview', children=[
        dcc.Tab(label='Overview', value='overview'),
        dcc.Tab(label='Backtest', value='backtest')
    ]),
    html.Div(id='tab-content')
])

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('stock-dropdown', 'value')
)
def render_tab(tab, ticker):
    df = fetch_data(ticker)
    if tab == 'overview':
        fig_amihud = go.Figure()
        fig_amihud.add_trace(go.Scatter(x=df.index, y=df['amihud'], mode='lines', name='Amihud Ratio'))
        fig_amihud.add_trace(go.Scatter(
            x=df[df['if_anomaly'] == -1].index,
            y=df[df['if_anomaly'] == -1]['amihud'],
            mode='markers', marker=dict(color='red', size=6), name='Anomaly'))
        fig_amihud.update_layout(title=f'{ticker} Amihud Ratio with Anomalies')

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume'))
        fig_vol.update_layout(title=f'{ticker} Daily Volume')

        return html.Div([
            kpi_cards(df),
            dcc.Graph(figure=fig_amihud),
            dcc.Graph(figure=fig_vol)
        ])

    elif tab == 'backtest':
        return html.Div([
            html.Br(),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=df.index.min().date(),
                max_date_allowed=df.index.max().date(),
                start_date=df.index[-60].date(),
                end_date=df.index[-1].date()
            ),
            html.Div(id='backtest-graph')
        ])

@app.callback(
    Output('backtest-graph', 'children'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('stock-dropdown', 'value')
)
def update_backtest(start_date, end_date, ticker):
    df = fetch_data(ticker)
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    df_range = df.loc[mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_range.index, y=df_range['return'].cumsum(), mode='lines', name='Cumulative Return'))
    fig.add_trace(go.Scatter(x=df_range.index, y=df_range['amihud'], mode='lines', name='Amihud Ratio', yaxis='y2'))
    fig.update_layout(
        title=f'{ticker} Backtest: Return vs Amihud',
        yaxis=dict(title='Cumulative Return'),
        yaxis2=dict(title='Amihud Ratio', overlaying='y', side='right')
    )
    return dcc.Graph(figure=fig)

if __name__ == '__main__':
    
import os
import dash
from dash import dcc, html

# 初始化 app
app = dash.Dash(__name__)
server = app.server  # 給 Render 用的 WSGI server 接入點

# 簡單首頁
app.layout = html.Div([
    html.H1("🐾 Hello from Render!"),
    dcc.Graph(
        figure={
            "data": [{"x": [1, 2, 3], "y": [4, 1, 2], "type": "bar"}],
            "layout": {"title": "Sample Graph"},
        }
    )
])

# ✅ 正確綁定 host & port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))  # Render 會自動設 PORT
    app.run_server(host="0.0.0.0", port=port)


