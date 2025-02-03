import streamlit as st
from PIL import Image


def app():
    st.title("How daily Stock allocation works")
    st.text('''
    Our app uses Deep Reinforcement Learning(DRL) agent for stock trading.
    Suppose that we have a well trained DRL agent “DRL Trader”, we want to use it to trade multiple stocks in our portfolio.
    1.Assume we are at time t, at the end of day at time t, we will know the open-high-low-close price of the Dow 30 constituents stocks. 
     We can use these information to calculate technical indicators such as MACD, RSI, CCI, ADX. 
     In Reinforcement Learning we call these data or features as “states”.
    2.We know that our portfolio value V(t) = balance (t) + dollar amount of the stocks (t).
    3.We feed the states into our well trained DRL Trader, the trader will output a list of actions, the action for each stock is a value within [-1, 1], 
     we can treat this value as the trading signal, 1 means a strong buy signal, -1 means a strong sell signal.
    4.We calculate k = actions *h_max, h_max is a predefined parameter that sets as the maximum amount of shares to trade. So we will have a list of shares to trade.
    5.The dollar amount of shares = shares to trade* close price (t).
    6.Update balance and shares. These dollar amount of shares are the money we need to trade at time t.
     The updated balance = balance (t) −amount of money we pay to buy shares +amount of money we receive to sell shares.
     The updated shares = shares held (t) −shares to sell +shares to buy.
    7.So we take actions to trade based on the advice of our DRL Trader at the end of day at time t 
     (time t’s close price equals time t+1’s open price). We hope that we will benefit from these actions by the end of day at time t+1.
    8.Take a step to time t+1, at the end of day, we will know the close price at t+1, the dollar amount of the stocks (t+1)= sum(updated shares * close price (t+1)).
     The portfolio value V(t+1)=balance (t+1) + dollar amount of the stocks (t+1).
    9.So the step reward by taking the actions from DRL Trader at time t to t+1 is r = v(t+1) − v(t). The reward can be positive or negative in the training stage.
     But of course, we need a positive reward in trading to say that our DRL Trader is effective.
    10.Repeat this process until termination.
    ''')
    st.subheader("Output results compared to particular stock(TCS) :")

    image1 = Image.open('img/1.png')
    st.image(image1, caption='Cumulative Returns Comparison')
    image1 = Image.open('img/2.png')
    st.image(image1, caption='Cumulative Returns on logarithmic scale')
    image1 = Image.open('img/3.png')
    st.image(image1, caption='Returns')
    image1 = Image.open('img/4.png')
    st.image(image1, caption='Rolling Volatility')
    image1 = Image.open('img/5.png')
    st.image(image1, caption='Roling Sharpe Ratio')
    image1 = Image.open('img/6.png')
    st.image(image1, caption='Top 5 Dropdown Periods')
    image1 = Image.open('img/7.png')
    st.image(image1, caption='Monthly Returns, Annual Returns, Monthly Returns Distribution, Return Quantiles (Daily,Weekly,Monthly)')


app()
