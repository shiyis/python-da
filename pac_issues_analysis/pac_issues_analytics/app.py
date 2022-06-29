import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input

data = pd.read_csv("snorkel_topics.csv")
data = data.dropna()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Avocado Analytics: Understand Your Avocados!"

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="🗳️", className="header-emoji"),
                html.H1(
                    children="PAC Issues Analytics", className="header-title"
                ),
                html.P(
                    children="Analyze and understand the political landscape through PAC candidate issues",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Name", className="menu-title"),
                        dcc.Dropdown(
                            id="name-filter",
                            options=[
                                {"label": name, "value": name}
                                for name in data.name.unique()
                            ],
                            value = 'ADERHOLT ROBERT B',
                            clearable=False,
                            className="dropdown",
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Issues", className="menu-title"),
                        dcc.Dropdown(
                            id="label-filter",

                            value="economy",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ],
                ),
            ],
            className="menu",
        ),
        html.Div([
            dcc.Tabs(id="tabs-with-classes", 
                value='tab-1',
                parent_className='custom-tabs', 
                className='custom-tabs-container',
                children=[
                dcc.Tab(label='Original Text', 
                        value='tab-1',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='Paraphrase',
                        className='custom-tab', 
                        selected_className='custom-tab--selected',
                        value='tab-2'),
            ]),

            html.Div(id='tabs-content-classes', className='tab-content')
            ]
        )
    ]
)

@app.callback(
    Output('label-filter', 'options'),
    [Input('name-filter', 'value')])
def set_label_options(selected_name):
    new_df = data.loc[data['name']==selected_name]
    return [{'label': i.upper(), 'value': i.upper()} for i in new_df['topics'].unique()]
    
@app.callback(
    Output('label-filter', 'value'),
    [Input('label-filter', 'options')])
def set_label_value(available_options):
    print(available_options[0]['value'])
    return available_options[0]['value']

@app.callback(
    [Output('tabs-content-classes', 'children')],
    [Input('name-filter', 'value'),
    Input('label-filter', 'value'),
    Input('tabs-with-classes', 'value')])
def render_content(selected_name, selected_label,tab):
    label = selected_label.lower()
    new_df = data.loc[data['name'] == selected_name.rstrip()]
    new_df2 = new_df.loc[data['topics'] == label]
    print(new_df2['paraphrase'])


    if tab == 'tab-1':
        return [html.Div([
            html.H3(''),
            html.P(children=new_df2['text'].apply(str).to_list(), className='text-content-1')
        ])]
    elif tab == 'tab-2':
        return [html.Div([
            html.H3(''),
            html.P(children=new_df2['paraphrase'].apply(str).to_list(),className='text-content-1')
        ])]



if __name__ == "__main__":
    app.run_server(debug=True)