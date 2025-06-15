sequence of data points - uniform time interval
smoothing, ARIMA
https://www.geeksforgeeks.org/time-series-analysis-and-forecasting/
## Assumptions
consecutive observation - equally spaced - time
index -> daily, weekly, monthly, yearly, ...
no missing values

## Types of Trend
Linear, rapid, periodic, varying variance

## Transformation
log - reduce rapid growth, stabilise variance
diff - remove linear/periodic trends
box-cox 

$$y = \frac{x^{\lambda - 1}}{\lambda}\ if\ \lambda \neq 0$$
$$y = log(x)\ if\ \lambda = 0$$

## Resampling
change time frequency

single data, avg of freq values

Downsampling
https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
```python
df.index = pd.DatetimeIndex( df.columnname )
downsampled = df.resample('Q').sum()
downsampled.index.rename('Quarter', inplace=True)
```

## Components of Time Series
Trend - increase/decrease
Seasonal - pattern regular interval
Cyclic
Random/Noise

## Decomposition
https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html

**additive**

$$y_{t} = \hat{T_{t}} + \hat{S_{t}} + \epsilon_{t}$$
**multiplicative**
$$y_{t} = \hat{T_{t}}  \hat{S_{t}}  \epsilon_{t}$$
![[Pasted image 20250507174721.png]]

## Forecasting

### Naive Forecast
next value <- last value

### Seasonal Naive Forecast
next value <- last value in similar period

### Moving Average
next value -> shifted mean value
#### Rolling mean
next value -> last n value means
## Visualisation

#### Centred Rolling Mean

## Forecasting Models
https://www.geeksforgeeks.org/exponential-smoothing-for-time-series-forecasting/

### Smoothing

![[Pasted image 20250507195551.png]]
![[Pasted image 20250507195604.png]]
- Simple or Single Exponential smoothing
- Exponential SmoothingExponentialSmoothing, 
	- Linear Smoothing
	- Double Exponential Smoothing
- Holt-Winters’ exponential smoothing
- Damping

### Exponential Smoothing
more weights to recent values
#### SimpleExpSmoothing
https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.SimpleExpSmoothing.html
#### Holt Forecasting Methods
Without Seasonality
- Linear
- Multiplicative
- Damped Linear
- Damped Multiplicative
```python
from statsmodels.tsa.api import Holt
ses=Holt(y_train, damped_trend=True, exponential=True)
fit4=ses.fit(smoothing_level=a, smoothing_trend=b, damping_trend=p)
```
#### Holt-Winters’ Forecasting Methods
https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html#statsmodels.tsa.holtwinters.ExponentialSmoothing
With Seasonality
- Linear
- Multiplicative
- Damped Linear
- Damped Multiplicative

```python
from statsmodels.tsa.api import ExponentialSmoothing
ses=ExponentialSmoothing(y_train, seasonal_periods = 12, trend='mul', seasonal='mul', damped_trend=True, exponential=True)
fit1=ses.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
```

**additive method - trend + seasonal**
level, trend, seasonal

**multiplicative method - e**


### Seasonality


## Stationary Process
A stationary time series is one whose statistical properties, like mean, variance, and auto-correlation, do not change over time

![[Pasted image 20250508161042.png]]

White Noise
### Random Walk Model
$$y_{t} = y_{t-1} + \epsilon_{t}$$
where epsilon is white noise

### Test to find Stationarity?
Augmented Dickey Fuller - ADF Test
https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html

ADF test is used to determine the presence of unit root in the series, and hence helps in understand if the series is stationary or not. The null and alternate hypothesis of this test are:
Null Hypothesis: The series has a unit root.
Alternate Hypothesis: The series has no unit root.
If the null hypothesis in failed to be rejected, this test may provide evidence that the series is non-stationary.
$$y_{t} = \beta y_{t-1} + \epsilon_{t}$$

```python
from statsmodels.tsa.stattools import adfuller
diff2 = diff1.diff()[1:]
results = adfuller(diff2, maxlag=10)
if results[1] < 0.05:
    print("Time Series is stationary")
else:
    print("Time Series is non-stationary")
```

## Auto correlation
https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_acf.html
Auto-correlation measures the degree of similarity between a given time series and the lagged version of that time series over successive time periods. 

$$AC_{lag_{1}} = Corr(Orig, Lag_{1})$$
$$\rho(k) = \frac{Cov(X_{t}, X_{t-k})}{\sigma(X_{t})\sigma(X_{t-k})}$$

```python
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(y, lags=20, alpha=None)
```

![[Pasted image 20250508170403.png]]

## Auto regressive Model
### ARIMA - Autoregressive Integrated Moving Average
https://www.geeksforgeeks.org/model-selection-for-arima/
https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
forecasting future values based on historical patterns within time series data

p,d,q
Autoregression (AR): This component relates the present value to its past values through a regression equation. represented by the **parameter p**.

Differencing (I for Integrated): It involves differencing the time series data to make it stationary, ensuring that the mean and variance are constant over time. represented by the **parameter d**

Moving Average (MA): This component uses the dependency between an observation and a residual error from a moving average model applied to lagged observations. represented by the **parameter q**

### Simple Moving Average Model

$$Y_{t} = c + \epsilon_{t}+\theta_{1}\epsilon_{t-1}+\theta_{2}\epsilon_{t-2}\dots +\theta_{q}\epsilon_{t-q}$$


### ARMA - Autoregressive Moving Average (ARMA)
arima model 

## AutoARIMA
https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html#pmdarima.arima.AutoARIMA

```python
pmdarima.arima.AutoARIMA
```


### SARIMAX

https://www.kaggle.com/code/jurk06/auto-arima-on-multivariate-time-series/notebook
pmdarima - https://pypi.org/project/pmdarima/
