import pandas as pd
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import math
import ta
from keras.models import load_model
from sklearn.model_selection import train_test_split

import yfinance as yf

start = '2009-01-01'
end = '2021-09-26'


def app():
    st.title("Invest With Ease")

    st.subheader("single stock prediction")
    tick = st.text_input("Enter stock name : ", 'RELIANCE.NS')
    dataset = yf.download(tick, start, end, progress=False)
    dataset = ta.add_all_ta_features(
        dataset, open="Open", high="High", low="Low", close="Close", volume="Volume")
    data = dataset[['Open',
                    'High',
                    'Low',
                    'Close',
                    'momentum_rsi',
                    'trend_macd_diff',
                    ]]
    st.subheader("Data described")
    st.write(data.describe())

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    sc_Y = MinMaxScaler()
    scaled_data = sc.fit_transform(data)
    sc_Y.fit(data['Close'].values.reshape(-1, 1))
    scaled_data = pd.DataFrame(
        scaled_data, columns=data.columns, index=data.index)

    st.subheader("Closing price vs time chart")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(dataset.Close)
    st.pyplot(fig)

    N_past = 100
    M_future = 7
    test_percentage = 0.15
    test_split = int(len(scaled_data)*(1 - test_percentage))
    train = scaled_data[:test_split]
    test = scaled_data[test_split-N_past:]

    def split_dataset_X_Y(data, past, future, jump=1):
        X, Y = [], []
        for i in range(0, len(data) - past - future + 1, jump):
            X.append(data[i: (i+past)].values)
            Y.append(data['Close'][(i + past): (i + past + future)])
        return np.array(X), np.array(Y)
    X_train, y_train = split_dataset_X_Y(train, N_past, M_future)
    X_test, y_test = split_dataset_X_Y(test, N_past, M_future)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.2, random_state=45)

    model = load_model('lstm_model2.h5')
    y_pred = model.predict(X_test)
    df = pd.DataFrame({'y_test': sc_Y.inverse_transform(y_test[:, 0].reshape(-1, 1)).flatten(),
                      'y_pred': sc_Y.inverse_transform(y_pred[:, 0].reshape(-1, 1)).flatten()},
                      test[N_past:N_past+len(y_test)].index)

    st.subheader('Test predictions: 1 day ahead')
    fig = plt.figure(figsize=(10, 4), dpi=150)
    df['y_test'].plot()
    df['y_pred'].plot()
    plt.legend(loc='best')
    plt.title('Test predictions: 1 day ahead')
    plt.grid(True)
    st.pyplot(fig)

    st.subheader('Test predictions: %i days ahead' % M_future)

    df_2 = pd.DataFrame({'y_test': sc_Y.inverse_transform(y_test[:, -1].reshape(-1, 1)).flatten(),
                         'y_pred': sc_Y.inverse_transform(y_pred[:, -1].reshape(-1, 1)).flatten()},
                        test[N_past+M_future-1:].index)
    fig = plt.figure(figsize=(10, 4), dpi=150)
    df_2['y_test'].plot()
    df_2['y_pred'].plot()
    plt.legend(loc='best')
    plt.title('Test predictions: %i days ahead' % M_future)
    plt.grid(True)
    st.pyplot(fig)

    train = scaled_data[(test_split-M_future-N_past) % M_future: test_split]
    test = scaled_data[test_split - N_past:]
    X_train, y_train = split_dataset_X_Y(train, N_past, M_future, M_future)
    X_test, y_test = split_dataset_X_Y(test, N_past, M_future, M_future)
    train_predictions = model.predict(X_train).ravel()
    test_predictions = model.predict(X_test).ravel()
    y_true = np.concatenate((y_train.ravel(), y_test.ravel()), axis=0)
    for _ in range(len(y_true)-len(train_predictions)):
        train_predictions = np.append(train_predictions, np.nan)

    for _ in range(len(y_true)-len(test_predictions)):
        test_predictions = np.insert(test_predictions, 0, np.nan, axis=0)
    assert (y_true.shape == train_predictions.shape == test_predictions.shape)
    lb = (test_split-M_future-N_past) % M_future + N_past
    ub = lb + len(y_true)
    complete_df = pd.DataFrame({'y_true': sc_Y.inverse_transform(y_true.reshape(-1, 1)).flatten(),
                                'train_predictions': sc_Y.inverse_transform(train_predictions.reshape(-1, 1)).flatten(),
                                'test_predictions': sc_Y.inverse_transform(test_predictions.reshape(-1, 1)).flatten()},
                               data[lb:ub].index)

    st.subheader('Predict %i days into the future from the last %i past days' % (
        M_future, N_past))
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=complete_df.index,
                             y=complete_df['y_true'],
                             name='True',
                             mode='lines',
                             line=dict(color='steelblue',
                                       width=2)))
    fig.add_trace(go.Scatter(x=complete_df.index,
                             y=complete_df['train_predictions'],
                             name='Predicted over Train',
                             mode='lines',
                             line=dict(color='darkorange',
                                       width=2)))
    fig.add_trace(go.Scatter(x=complete_df.index,
                             y=complete_df['test_predictions'],
                             name='Predicted over Test',
                             mode='lines',
                             line=dict(color='green',
                                       width=2)))
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            gridcolor='rgb(204, 204, 204)',
            showticklabels=True,
            linecolor='black',
            linewidth=1,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            )),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgb(204, 204, 204)',
            linewidth=0.3,
            linecolor='black',
            zeroline=True,
            showline=True,
            showticklabels=True,
        ),
        title={
            'text': 'Predict %i days into the future from the last %i past days' % (M_future, N_past),
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis_title='Price',
        xaxis_title='Date',
        autosize=False,
        width=1040,
        height=608,
        plot_bgcolor='white')
    st.plotly_chart(fig)
