import os
import pandas as pd
import yfinance as yf
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from flask import Flask

# -- 讀取資料 --
df = pd.read_csv("dataset/US_stock_data.csv")

# -- 初始化 Flask 與 Dash --
server = Flask(__name__)
app = Dash(__name__, server=server)

# -- 建立下拉選單選項 --
stock_options = [{'label': symbol, 'value': symbol} for symbol in df['symbol'].unique()]

# -- App Layout --
app.layout = html.Div([
    html.H1("📈 US Stock Dashboard", style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='stock-dropdown',
        options=stock_options,
        value=stock_options[0]['value'],
        clearable=False,
        style={'width': '60%', 'margin': '0 auto'}
    ),
    dcc.Graph(id='price-graph'),
    html.Div(id='table-container')
])

# -- 回傳圖與表 --
@app.callback(
    [Output('price-graph', 'figure'),
     Output('table-container', 'children')],
    [Input('stock-dropdown', 'value')]
)
def update_dashboard(selected_stock):
    filtered_df = df[df['symbol'] == selected_stock]

    figure = {
        'data': [
            go.Scatter(
                x=filtered_df['date'],
                y=filtered_df['close'],
                mode='lines+markers',
                name='Close Price'
            )
        ],
        'layout': go.Layout(
            title=f'{selected_stock} Stock Price',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'}
        )
    }

    table = dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in filtered_df.columns],
        data=filtered_df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=10
    )

    return figure, table

# -- Render 執行區（注意縮排！）--
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
