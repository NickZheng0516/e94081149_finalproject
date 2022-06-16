# %%
# -*- coding: utf-8 -*-
# 載入需要的套件
# 科學計算、數據處理常用的兩個基礎套件
import numpy as np
import pandas as pd

# 作圖、視覺化的套件
import matplotlib.pyplot as plt
# Yahoo finance 套件，用來下載股價資訊
import yfinance as yf
import datetime

# SciKit-Learn (sklearn) 套件，用來快速、簡單建構基礎模型的套件
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# tkinter與matplotlib整合套件
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# %%
# 設定想要的股票代碼資訊，以及要下載股價資訊的時間範圍
def process_data():
    # 處理輸入
    stock_name = text_1.get('1.0', 'end')
    start = text_2.get('1.0', 'end')
    start = start.split('-')
    start = datetime.datetime(int(start[0]), int(start[1]), int(start[2]))
    end = int(text_3.get('1.0', 'end'))
    end = datetime.timedelta(days=int(end))
    end = start + end

    # 下載股價資訊
    df_full = yf.download(stock_name, start, end, interval='1d').dropna()
    # 畫出股價資訊
    df_raw = df_full[df_full.columns[-1]]
    fig1 = plt.figure(figsize=(8, 4))
    sub1 = fig1.add_subplot(1,1,1)
    sub1.plot(df_raw.index, df_raw, label=stock_name)
    sub1.grid()
    sub1.legend()
    canvas1=FigureCanvasTkAgg(fig1, win)
    canvas1.get_tk_widget().grid(row=12, column=0)

    # 畫出移動平均線，每 20 天為一個移動單位
    window = 20
    df_MA = df_full[df_full.columns[-1]].rolling(window).mean()
    # X 只拿第 1 天到第 N-1 天，而 y 則取第 2 天到第 N 天
    # X 只拿第 1 天到第 N-1 天，而 y 則取第 2 天到第 N 天
    df_X = df_full.iloc[:-1,:-1]
    df_y = df_full.iloc[1:,-1]
    X = df_X.to_numpy() 
    y = df_y.to_numpy() 
    # 訓練/測試的資料分割，以前 80% 的天數資料做訓練，後 20% 來做測試
    num_data = df_X.shape[0]
    split_ratio = 0.8
    ind_split = int(split_ratio * num_data)
    X_train = X[:ind_split]
    y_train = y[:ind_split].reshape(-1,1)
    X_test = X[ind_split:]
    y_test = y[ind_split:].reshape(-1,1)
    split_time = df_X.index[ind_split]
    reg_linear = LinearRegression()
    reg_linear.fit(X_train, y_train)
    # 將訓練好的模型，用來做預測
    trainings = reg_linear.predict(X_train).reshape(-1,1)
    predictions = reg_linear.predict(X_test).reshape(-1,1)
    # 將預測結果合再一起
    all_pred = np.concatenate((trainings, predictions), axis=0)
    # 計算方均根差
    train_rmse = mean_squared_error(trainings, y_train, squared=False)
    test_rmse = mean_squared_error(predictions, y_test, squared=False)
    print(f'Training RMSE is: {train_rmse}')
    print(f'Testing RMSE is: {test_rmse}')

    # 將預測和真實的股價，放進 df_linear 以便做圖
    df_linear = pd.DataFrame(all_pred, columns=['Linear '+df_full.columns[-1]], index=df_y.index)
    df_linear[df_full.columns[-1]] = y
    # 畫出結果
    fig2=plt.figure(figsize=(8, 4))
    sub2 = fig2.add_subplot(1, 1, 1)
    sub2.plot(df_linear.index, df_linear.iloc[:, 0], label=stock_name, color='r')
    sub2.plot(df_linear.index, df_linear.iloc[:, 1], label=stock_name, color='C0')
    sub2.grid()
    sub2.legend()
    sub2.axvline(pd.Timestamp(split_time),color='orange')
    canvas2=FigureCanvasTkAgg(fig2, win)
    canvas2.get_tk_widget().grid(row=20, column=0)

# %%
# 建立一個視窗
win = tk.Tk()
win.title('台灣股票行情查詢')
win.geometry('1600x900')

# 印出第一列敘述與輸入框
label_1 = tk.Label(win, text='台灣股票代碼：', font=('Arial', 12))
label_1.grid(row=0, column=0, rowspan=3, sticky=tk.W+tk.N+tk.S)
text_1 = tk.Text(win, height=3, width=30)
text_1.grid(row=0, column=1, rowspan=3, sticky=tk.W)

# 印出第二列敘述與輸入框
label_2 = tk.Label(win, text='預估起始時間(yyyy-mm-dd)：', font=('Arial', 12))
label_2.grid(row=3, column=0, rowspan=3, sticky=tk.W+tk.N+tk.S)
text_2 = tk.Text(win, height=3, width=30)
text_2.grid(row=3, column=1, rowspan=3, sticky=tk.W)

# 印出第三列敘述與輸入框
label_3 = tk.Label(win, text='持續時間(天)：', font=('Arial', 12))
label_3.grid(row=6, column=0, rowspan=3, sticky=tk.W+tk.N+tk.S)
text_3 = tk.Text(win, height=3, width=30)
text_3.grid(row=6, column=1, rowspan=3, sticky=tk.W)

# 印出按鈕
button = tk.Button(win, height=3, text='開始計算', command=process_data)
button.grid(row=9, column=0, rowspan=3, columnspan=2, sticky=tk.N+tk.E+tk.W)

win.mainloop()
