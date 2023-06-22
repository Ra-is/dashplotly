from jupyter_dash import JupyterDash
from dash import dcc, html,dash_table
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import date
from prophet import Prophet

df = pd.read_csv("customerData.csv",low_memory=False)
df['time_created'] = pd.to_datetime(df['time_created'], format="%Y-%m-%dT%H:%M:%S.%fZ")
df = df.assign(year=df['time_created'].dt.year,
               month=df['time_created'].dt.month,
               day=df['time_created'].dt.day)
new_columns = [
    'time_created',
    'year',
    'month',
    'day',
    'debit_amt',
    'credit_amt',
    'recipient_wallet_number',
    'sender_wallet_number',
    'transaction_type',
    'time_completed'
    
]
df = df.reindex(columns=new_columns)
df.drop(columns=['time_completed'],inplace=True)

load_figure_template("CERULEAN")

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])
PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search")),
        dbc.Col(
            dbc.Button(
                "Search", color="primary", className="ms-2", n_clicks=0
            ),
            width="auto",
        ),
    ],
    className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)
card_content = [
    dbc.CardHeader(id="TotalTransactionTitle"),
    dbc.CardBody(
        [
            html.H5(id="TotalTransaction", className="card-title"),
        ]
    ),
]
card_stats = dbc.Row(
    [
        dbc.Col(dbc.Card(card_content, color="primary", outline=True), width=3),
    ],
    className="mb-4",
)

app.layout = html.Div(
    children=[
    dbc.Navbar(
    dbc.Container(
        style={"padding": "10px"},
        children=[
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(
                         html.Img(src=PLOTLY_LOGO, height="30px")
                        ),
                        
                        dbc.Col(
                         dbc.NavbarBrand(children="Analytical Dashboard", id="navtext", className="ms-2")
                        ),
                        
                    ],
                    align="center",
                    className="g-0",
                ),
                href="",
                style={"textDecoration": "none"},
            ),
           
        ]
    ),
    color="red",
    dark=True,
    ),
    html.Div(
    style={"margin": "10px"},
     
    children=[
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Generale Observation", 
                    tab_id="tab-1",
                    children=[
                    html.Br(),
                    dbc.Row([
                    dbc.Col(
                     # we put radio item here
                     dcc.RadioItems(
                        options=["Sum", "Average"], 
                        value="Sum",
                        id="Metric-Radio",
                        style={"fontSize": 20},
                        labelStyle={"padding-right": "10px"},
                        className="dbc"
                    ),
                    width=2
                    ),
                    dbc.Col(card_stats,width=10)
                ]),
    dbc.Row([
        # we put one graph here bar char
        dbc.Col(dcc.Graph(id="TransactionTypeByMonth"),width=8),
        dbc.Col(dcc.Graph(id="TransactionTypePercentage"), width=4),
    ]),
    html.Br(),
    dbc.Row([
        # churn customers
        dbc.Col(dcc.Graph(id="UniqueCustomers"),width=6),
            dbc.Col(
            dbc.Row([
                html.H5('Unique Customers Count',style={'text-align': 'center', 'color':'grey'}),
                html.Br(),
                html.Div(),
                html.Div(id="uniqueCustomersTable")
            ]),
            width=6
            ),
    ]),
    html.Div(),
                    ],
                ),
                dbc.Tab(
                    label="Forecasting",
                    tab_id="tab-2",
                    children=[
#                        html.H5('this is tabl two'),
                        dbc.Row([
                             dbc.Col(html.Div(),width=4),
                             dbc.Col(html.H5(id="forecastTitle"),width=4),
                             dbc.Col(html.Div(),width=4),
                        ]),
                       dbc.Row([
                         dbc.Col(
                     # we put radio item here
                     dcc.RadioItems(
                        options=["Graph"], 
                        value="Graph",
                        id="Forecast-Radio",
                        style={"fontSize": 20},
                        labelStyle={"padding-right": "10px"},
                        className="dbc"
                    ),
                    width=2
                    ),
                        dbc.Col(
                        dcc.Graph(id="forecastTrend"),
                        width=5
                        ),
                        dbc.Col(
                        dcc.Graph(id="forecastWeekly"),
                        width=5
                        )
                       ]),
                        dbc.Row([
                             dbc.Col(html.Div(),width=4),
                            dbc.Col(
                                html.Div(id="forecastTable"),
                                 width=4
                            ),
                            dbc.Col(html.Div(),width=4)
                             
                        ])
                        
                    ],
                ),
            ],
            id="ta",
            active_tab="tab-1",
        ),     
    ]
    ),
    
])


               
@app.callback(
    Output("TotalTransactionTitle", "children"),
    Output("TotalTransaction", "children"),
    Input('Metric-Radio', 'value')
)
def calculateTotalTransaction(value):
    if not value:
        raise PreventUpdate
    if value == "Sum":
        formatted_number = "{:,.2f}".format(df.debit_amt.sum(), 2)
        debit_sum = f"GHC {formatted_number}"
        return "Total of all Debit Transaction", debit_sum
    else:
        formatted_number = "{:,.2f}".format(df.debit_amt.mean(), 2)
        debit_average = f"GHC {formatted_number}"
        return "Average of all Debit Transaction", debit_average

@app.callback(
    Output("TransactionTypeByMonth", "figure"),
    Output("TransactionTypePercentage", "figure"),
    Input('navtext', 'children')
)
def plot_bar_pie_chat(value):
    transaction_amount_month_bar = df.groupby(['month','transaction_type'],as_index=False).agg({
    'debit_amt': ['sum']
    })
    transaction_amount_month_bar.columns = ['month', 'transaction_type', 'debit_amt'] 
    fig_bar = px.bar(
        transaction_amount_month_bar,
        x='month',
        y='debit_amt',
        color='transaction_type',
        barmode='group'
    ).update_layout(
    
    title_font=dict(
        color="grey",
        size=20
    ),
    title={
        'text': "Total Transaction Types By Month",
        "x": .50,
        "y": .95,
        "xanchor": "center"
    },
    legend_title="",
    width=750
    )
    
    df.groupby(['transaction_type']).agg({
    'debit_amt': ['sum']
    })
    transaction_amount_month = df.groupby(['transaction_type'],as_index=False).agg({
    'debit_amt': ['sum']
    })
    transaction_amount_month.columns = ['transaction_type','debit_amt_sum']
    fig_pie = px.pie(
     transaction_amount_month,
     values="debit_amt_sum",
     names="transaction_type",
    hole=.7
    ).update_layout(
    
    title_font=dict(
        color="grey",
        size=20
    ),
    title={
        'text': "Total Transaction Types By Percentages",
        "x": .50,
        "y": .95,
        "xanchor": "center"
    },
    legend_title="",
    width=750
    )

    return  fig_bar, fig_pie
@app.callback(
    Output("UniqueCustomers", "figure"),
    Output("uniqueCustomersTable","children"),
     Input('navtext', 'children')
)

def uniqueCustomerUpdate(value):
    user_count_mnth = df.groupby('month')['sender_wallet_number'].nunique().reset_index(name='count')
    fig = px.line(user_count_mnth, x='month', y='count', 
    labels={
    'month': 'Month',
            'count': 'Counts',
        }).update_xaxes(
            title="Month",
        ).update_yaxes(
            showticksuffix="last",
        ).update_layout(

            title_font=dict(
                color="grey",
                size=20
            ),
            title={
                'text': "Number of Unique Sender Wallet Numbers by Month",
                "x": .50,
                "y": .95,
                "xanchor": "center"
            },
            legend_title="",
            width=750
        )
    data_tables = dbc.Table.from_dataframe(
        user_count_mnth,
        striped=True, 
        bordered=True, 
        hover=True,
#         color="dark",
        class_name="dbc"
    )
    return fig, data_tables

@app.callback(
    Output("forecastTitle", "children"),
    Output("forecastTrend", "figure"),
    Output("forecastWeekly", "figure"),
    Output("forecastTable", "children"),
    Input('Forecast-Radio', 'value')
)

def myForeCast(value):
    if not value:
        raise PreventUpdate
    if value == 'Graph':
        new_df = df.loc[:,['time_created','debit_amt']]
        new_df['date'] = new_df['time_created'].dt.date
        new_df.drop('time_created',axis=1,inplace=True)
        d = new_df.groupby(['date']).agg(
            debit_amount=('debit_amt','sum')
            )
        d['date'] = d.index
        d.columns = ['y', 'ds']
        m = Prophet()
        m.fit(d)
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend'))
        fig.update_layout(title='Components of Forecast')
        
        # weekly
        weekly_by_day = forecast.groupby(forecast['ds'].dt.dayofweek)['weekly'].mean()

        # create bar chart of weekly component by day of the week using Plotly Express
        weekly_fig = px.line(x=weekly_by_day.index, y=weekly_by_day.values, title='Weekly Component by Day of Week')
        weekly_fig.update_layout(xaxis_tickvals=[0, 1, 2, 3, 4, 5, 6], xaxis_ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

        forecast.set_index('ds',inplace=True,drop=False)
        forecast_values = forecast['yhat'].apply(lambda x: "{:,.2f}".format(x, 2)).reset_index()
        forecast_values.columns = ['date','forecasted values']
        forecast_values.set_index('date',inplace=True, drop=False)
        forecast_values = forecast_values[forecast_values.date.apply(lambda x: x.to_pydatetime().date() > d.index.max() )]
        
        r = forecast_values.sort_index(ascending=True)
       
        
        data_tables = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in r.columns],
        data=r.to_dict("records"),
        sort_action="native",
        export_format="csv",
        page_size=10,
        )
        
        return "Forecasting",fig, weekly_fig, data_tables
    

app.run_server(debug=True, port=3055)