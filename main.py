import numpy as np
import pandas as pd
import math
import altair as alt 
from prophet import Prophet
from prophet.diagnostics import cross_validation 
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.plot import plot_plotly, plot_components_plotly

from PIL import Image
import streamlit as st 
import base64 #to open .gif files in streamlit app
from pathlib import Path 
from datetime import date, datetime 
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error 
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose 


pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 500) 
pd.set_option('display.width', 800) 
st.set_option('deprecation.showPyplotGlobalUse', False) 

def img_to_bytes(img_path):
	img_bytes = Path(img_path).read_bytes() 
	encoded = base64.b64encode(img_bytes).decode() 
	return encoded 

# functions
@st.experimental_memo
def get_simple_model(df):
	# df.columns = ['ds','y']
	m = Prophet(interval_width=0.95)
	m.fit(df) 
	return m 
	
@st.experimental_memo(ttl=60 * 60 * 24)
def comparison_chart(data, title):
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title=title)
        .mark_line()
        .encode(
            x="date",
            y='sales:Q',
            color="set",
        )
    )
    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="date",
            y='sales:Q',
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip('sales:Q', title=["Sales ($)"]),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()

@st.experimental_memo
def get_fc_charts(df, start_date, periods):
	st.write('#### Model predictions')
	df.columns = ['ds','y']
	m = get_simple_model(df)
	future = m.make_future_dataframe(periods=periods, freq = 'MS')
	forecast = m.predict(future)

	# main chart
	fig = plot_plotly(m, forecast)
	fig.update_layout(hovermode="x unified")
	fig.update_yaxes(title_text='Sales ($)')
	end_date = datetime.strftime(max_data + relativedelta(months=periods), '%Y-%m-%d').replace(' 0', ' ')
	fig.update_xaxes(title_text = 'Order Date',range=[start_date,end_date])
    #component chart
	fig_comp = plot_components_plotly(m,forecast)
	fig_comp.update_yaxes(title_text='Sales ($)')
	fig_comp.update_xaxes(title_text = 'Order Date')
	st.write(fig, fig_comp)
	# evaluate the predictions
	st.write('#### Evaluate model predictions')
	dates = df['ds'][-periods:].values
	y_true = df['y'][-periods:].values
	y_pred = forecast['yhat'][-periods:].values
	source = pd.DataFrame({'date':dates, 'Actual':y_true,'Predict':y_pred})
	source.set_index('date', inplace=True)
	source = source.reset_index().melt('date', var_name='set', value_name='sales')
	# plot
	chart = comparison_chart(source, "Simple - Prophet's prediction evaluation")
	st.altair_chart((chart).interactive(), use_container_width=True)
	# assess the model with MAE
	mae = mean_absolute_error(y_true, y_pred)
	st.success('MAE: %.3f' % mae)
	 
# get master data predictSuperstoreProfit/Sample - Superstore.csv
data = pd.read_csv('Sample - Superstore.csv', parse_dates=['Order Date', 'Ship Date'],encoding= 'unicode_escape').sort_values(by = ['Order Date','Order ID'])

# navigation dropdown
app_mode = st.sidebar.selectbox('Select Page',['Background','Superstore Performance'])

# page 1 
if app_mode == 'Background':
	# timeline image 
	p1_img1 = Image.open('images/timeline.png')
	st.image(p1_img1, width=700)
	st.write(' ')
	st.title('🛍️ Welcome to Superstore\'s predictive dashboard!')
	st.markdown(
			"""
		[<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/bichpham102/PredAnalyticsApp_2) <small> app-predictive-analytics 1.0.0 | Nov 2022</small>""".format(
			img_to_bytes("./images/github.png")
		),
		unsafe_allow_html=True,
		)
	st.markdown('---')

	st.subheader(" ")
	st.subheader('🎯 Objectives') 
	st.write('Superstore is a major Retail chain, whose stores are located across the US. In order to stay competitive in such a developed market, the company needs to be able to understand what has happened and what are likely to happen next to their business for prompt adjustments.')
	
	st.subheader(" ")
	st.subheader('🎯 Dataset') 
	st.write('What Superstore have available is their Sales data at Order Item level from 2014 to 2017. A preview of the dataset is shown below:')
	st.write(data.tail())
	st.write('Source: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final')

	st.subheader(" ")
	st.subheader('🎯 Approach') 
	#
	st.write('The dashboard is divided into 2 main tabs: ')
	st.markdown('#### 1 - Current Performance')
	
	st.markdown('This section shows Superstore\'s performance in terms of Monthly Sales overtime and from different perspectives, including:')
	st.markdown('* Performance Overview')
	st.markdown('* Performance by Segment')
	st.markdown('* Performance by Category')
	st.markdown('Each corresponds to a sub-tab, which contains (a) a breakdown of the time series into components and \
			(b) sales contribution by Segment/Category. These information is helpful for management to understand past performance, \
			and identify segment/category that have high risks or highly promising opportunity.')

	#
	st.markdown('#### 2 - Forecast')
	st.markdown('This section show the Monthly Sales forecast in the next 24 months that Superstore can expect. The predictions are done using \
			the Prophet library for time series forecasting for its simplicity, yet still able to yield good result compared to other alternatives. ')
	st.markdown("* Data is first process to the form required by Prophet with 2 columns only: 'ds' for date and 'y' for value")
	st.markdown('* A simple automated Prophet model is used first.')
	st.markdown('* If the results of the simple automated Prophet model look like it can be improved further, then we consider Parameters Tuning.')
	st.markdown('* Based on the best parameters determined in the previous step, we create another Prophet model and get new predictions.')

# page 2
if app_mode == 'Superstore Performance':
	p2_img1 = Image.open('images/timeline2.jpeg')
	st.image(p2_img1, width=700)
	st.write(' ')
	st.title(' 📊 Superstore\'s Performance Prediction')
	st.markdown('---')

	# inputs 
	max_data = max(data['Order Date'])
	min_data = min(data['Order Date'])
	start_date = st.date_input(
		'Select start date' 
		,min_data 
		,min_value = min_data
		,max_value = max_data
		)
	periods = st.slider('Pick a period to forcast (months)', 0, 48,24)

	st.subheader(" ")

	# GET DATA  
	data['Order Date'] = pd.to_datetime(data['Order Date'].apply(lambda x: x.date()) )

		## OVV
	df_ovv_1 = data[['Order Date','Sales']]
	df_ovv_2 = df_ovv_1.set_index('Order Date')
	df_ovv_2 = df_ovv_2.resample('MS').sum()
	df_ovv_3 = df_ovv_2.reset_index().rename({'Order Date':'ds','Sales':'y'})
		## SEG
	df_seg_1 = data[['Order Date','Segment','Sales']]
	df_seg_2 = df_seg_1.groupby(['Order Date','Segment'], as_index=True)['Sales'].sum() 
	df_seg_3 = pd.DataFrame(df_seg_2.unstack(level = 1)) 
	df_seg_3 = df_seg_3.resample('MS').sum()
	df_seg_pie = df_seg_1.groupby(['Segment'], as_index=True)['Sales'].sum() 
	max_seg_y = round(max(df_seg_3.max())+1000,-3)
		## CAT
	df_cat_1 = data[['Order Date','Category','Sales']]
	df_cat_2 = df_cat_1.groupby(['Order Date','Category'], as_index=True)['Sales'].sum() 
	df_cat_3 = pd.DataFrame(df_cat_2.unstack(level = 1)) 
	df_cat_3 = df_cat_3.resample('MS').sum()
	df_cat_pie = df_cat_1.groupby(['Category'], as_index=True)['Sales'].sum() 
	max_cat_y = round(max(df_cat_3.max())+1000,-3)

	


	# first main plot 
	st.write(' ')
	st.markdown('#### Superstore\'s Monthly Sales Value in $')
	plt.figure(figsize=(11,4))
	df = df_ovv_2[df_ovv_2.index > pd.to_datetime(start_date)].reset_index()
	c = alt.Chart(df).mark_line().encode(x='Order Date', y='Sales', tooltip=['Order Date', 'Sales'])
	st.altair_chart(c, use_container_width=True)


	st.write(' ')
	st.markdown('#### Details')
	tab1, tab2 = st.tabs(["Current Performance", "Forecast"])
	with tab1:
		tab11, tab12, tab13 = st.tabs(["by Component", "by Segment", "by Category"])
		with tab11:  # decompose
			plt.rc('figure',figsize=(14,8))
			plt.rc('font',size=15)
			result = seasonal_decompose(df_ovv_2,model='additive')
			fig = result.plot()
			st.pyplot()
		with tab12: # segment
			# pie chart
			st.write(f'Sales value contribution by Segment from {start_date} to {datetime.date(max_data)}. ')
			fig1, ax1 = plt.subplots()
			ax1.pie(df_seg_pie,  labels=df_seg_pie.index, autopct='%1.1f%%', startangle=90)
			ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
			st.pyplot(fig1)
			st.write(' ')
			# time series subplots
			st.write(f'Daily Sales value by Segment from {start_date} to {datetime.date(max_data)}. ')
			cols_plot = list(df_seg_3.columns)
			axes = df_seg_3[cols_plot].plot(subplots=True)
			for ax in axes:
				ax.set_ylabel('Daily Sales value in $') 
				ax.set_ylim(0, max_seg_y)
			st.pyplot()
		with tab13: # categories
			# pie chart
			st.write(f'Sales value contribution by Category from {start_date} to {datetime.date(max_data)}. ')
			fig1, ax1 = plt.subplots()
			ax1.pie(df_cat_pie,  labels=df_cat_pie.index, autopct='%1.1f%%', startangle=90)
			ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
			st.pyplot(fig1)
			st.write(' ')
			# time series subplots
			st.write(f'Daily Sales value by Category from {start_date} to {datetime.date(max_data)}. ')
			cols_plot = list(df_cat_3.columns)
			axes = df_cat_3[cols_plot].plot(subplots=True)
			for ax in axes:
				ax.set_ylabel('Daily Sales value in $') 
				ax.set_ylim(0, max_cat_y)
			st.pyplot()

	

	with tab2: #forecast
		tab21, tab22, tab23 = st.tabs(['Total Superstore','by Segment', 'by Category'])
		
		with tab22: # segment
			# create the model 
			st.write("Forecast use a Simple automated Prophet model")
			segments = data.Segment.unique()
			segment_forecast = st.selectbox("Select a Segment:", segments)
			df = df_seg_3[segment_forecast].reset_index()
			get_fc_charts(df, start_date=start_date,periods=periods)

		with tab23: # category
			st.write("Forecast use a Simple automated Prophet model")
			categories = data.Category.unique()
			category_forecast = st.selectbox("Select a Category:", categories)
			df = df_cat_3[category_forecast].reset_index()
			get_fc_charts(df, start_date=start_date,periods=periods)

		with tab21: # overall 
			df = df_ovv_2.reset_index()
			df.columns = ['ds','y']

			with st.expander("Simple automated Prophet model"):
				get_fc_charts(df, start_date=start_date,periods=periods)
			
			st.success('Simple automated Prophet model produces fairly good predictions but there are quite a gap vs. reality in the first 6 months predicted.\
				So we also consider Tuning the parameters to see if we can get better resutls. ')
			st.write(' ')

			# Tuning
			with st.expander("Tuning params code"):
				snippet =  '''
param_grid = {  
	'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
	'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
	}
# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here
# Use cross validation to evaluate all parameters
for params in all_params:
	m = Prophet(**params).fit(df)  # Fit model with given params
	df_cv = cross_validation(m, horizon=12, parallel="processes")
	df_p = performance_metrics(df_cv, rolling_window=1)
	rmses.append(df_p['rmse'].values[0])
# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
best_params = all_params[np.argmin(rmses)]
'''
				code_header_placeholder = st.empty() 
				snippet_placeholder = st.empty() 
				snippet_placeholder.code(snippet) 
				st.write('Best params: ')
				st.json({
							"changepoint_prior_scale": 0.001,
							"seasonality_prior_scale": 0.1
							})


			# create prophet model 
			st.write('#### Model predictions')
			mae = [] 
			m = Prophet( seasonality_mode = 'multiplicative'
						,seasonality_prior_scale = 0.1
						,changepoint_prior_scale = 0.001
						)
			m.fit(df)
			# forecasting 
			future = m.make_future_dataframe(periods=periods, freq = 'MS')
			forecast = m.predict(future)

			# main chart
			fig = plot_plotly(m, forecast)
			fig.update_layout(hovermode="x unified")
			fig.update_yaxes(title_text='Sales ($)')
			end_date = datetime.strftime(max_data + relativedelta(months=periods), '%Y-%m-%d').replace(' 0', ' ')
			fig.update_xaxes(title_text = 'Order Date',range=[start_date,end_date])
			#component chart
			fig_comp = plot_components_plotly(m,forecast)
			fig_comp.update_yaxes(title_text='Sales ($)')
			fig_comp.update_xaxes(title_text = 'Order Date')
			st.write(fig, fig_comp)
			# evaluate the predictions
			st.write('#### Evaluate model predictions')
			dates = df['ds'][-periods:].values
			y_true = df['y'][-periods:].values
			y_pred = forecast['yhat'][-periods:].values
			source = pd.DataFrame({'date':dates, 'Actual':y_true,'Predict':y_pred})
			source.set_index('date', inplace=True)
			source = source.reset_index().melt('date', var_name='set', value_name='sales')
			# plot
			chart = comparison_chart(source, "Tuned - Prophet's prediction evaluation")
			st.altair_chart((chart).interactive(), use_container_width=True)
			# assess the model with MAE
			mae = mean_absolute_error(y_true, y_pred)
			st.success('MAE: %.3f' % mae)

			st.success('Tuned model produces better fit results than an automated one, as indicated by a lower MAE.')

			
