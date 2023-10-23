import streamlit as st
from sportsipy.nfl.boxscore import Boxscores, Boxscore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hvplot.pandas
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

pd.set_option('display.max_columns', None)

def main():
    st.title('NFL Website')
    menu = ["Home", "NFL Prediction", "Stats"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Welcome to the NFL Website")

    elif choice == "NFL Prediction":
        nfl_prediction_tab()

def nfl_prediction_tab():
    st.subheader("NFL Game Prediction")

    # Let user select the prediction method
    method = st.radio("Select a prediction method", ["Logistic Regression", "Neural Network", "Random Forest"])

    pred_week = st.sidebar.slider("Select week for prediction", 3, 14)

    # Load and preprocess data
    df, X_test, test_df = prepare_data(pred_week)

    if method == "Logistic Regression":
        y_pred_unscaled, y_pred_scaled = logistic_regression_predict(df, X_test, pred_week)
        st.write("Logistic Regression - Unscaled\n")
        display_results(y_pred_unscaled, test_df)
        st.write("\nLogistic Regression - Scaled\n")
        display_results(y_pred_scaled, test_df)

    elif method == "Neural Network":
        y_pred = neural_network_predict(df, X_test, pred_week)
        st.write("Neural Network Predictions\n")
        display_results(y_pred, test_df)

    elif method == "Random Forest":
        y_pred = random_forest_predict(df, X_test, pred_week)
        st.write("Random Forest Predictions\n")
        display_results(y_pred, test_df)

def prepare_data(pred_week):
    df = pd.read_csv(r"C:\Users\ander\OneDrive\Desktop\2021_week_2_through_14.csv")
    comp_games_df = df[df['week'] < pred_week]
    pred_games_df = df[df['week'] == pred_week]
    train_df = comp_games_df
    test_df = pred_games_df
    X_test = test_df.drop(columns=['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    return df, X_test, test_df


def logistic_regression_predict(df, X_test, pred_week):
    comp_games_df = df[df['week'] < pred_week]
    train_df = comp_games_df
    X_train = train_df.drop(columns = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    y_train = train_df[['result']]

    # For unscaled predictions
    clf_unscaled = LogisticRegression()  
    clf_unscaled.fit(X_train, np.ravel(y_train.values))
    y_pred_unscaled = clf_unscaled.predict_proba(X_test)
    y_pred_unscaled = y_pred_unscaled[:, 1]

    # For scaled predictions
    scaler = StandardScaler()
    X_scaler = scaler.fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    clf_scaled = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True,
                                   intercept_scaling=1, class_weight='balanced', random_state=None,
                                   solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0)
    clf_scaled.fit(X_train_scaled, np.ravel(y_train.values))

    y_pred_scaled = clf_scaled.predict_proba(X_test_scaled)
    y_pred_scaled = y_pred_scaled[:, 1]

    return y_pred_unscaled, y_pred_scaled


def neural_network_predict(df, X_test, pred_week):
    comp_games_df = df[df['week'] < pred_week]
    train_df = comp_games_df
    X_train = train_df.drop(columns = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    y_train = train_df[['result']]

    number_input_features = X_train.shape[1]
    hidden_nodes_layer1 = (number_input_features + 1) // 2
    hidden_nodes_layer2 = (hidden_nodes_layer1 + 1) // 2
    hidden_nodes_layer3 = (hidden_nodes_layer2 + 1) // 2

    nn = Sequential()
    nn.add(Dense(units=hidden_nodes_layer1, activation='relu', input_dim=number_input_features))
    nn.add(Dense(units=hidden_nodes_layer2, activation='relu'))
    nn.add(Dense(units=hidden_nodes_layer3, activation='relu'))
    nn.add(Dense(units=1, activation='sigmoid'))
    nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    fit_model = nn.fit(X_train, y_train, epochs=500, verbose=0)

    y_pred = nn.predict(X_test)
    return y_pred

def random_forest_predict(df, X_test, pred_week):
    comp_games_df = df[df['week'] < pred_week]
    train_df = comp_games_df
    X_train = train_df.drop(columns = ['away_name', 'away_abbr', 'home_name', 'home_abbr', 'week', 'result'])
    y_train = train_df[['result']]

    rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
    rf_fitted = rf_model.fit(X_train, np.ravel(y_train.values))

    y_pred = rf_fitted.predict_proba(X_test)
    y_pred = y_pred[:, 1]

    return y_pred


def display_results(y_pred, test_df):
    for g in range(len(y_pred)):
        win_prob = int(y_pred[g] * 100)
        away_team = test_df.reset_index().drop(columns = 'index').loc[g,'away_name']
        home_team = test_df.reset_index().drop(columns = 'index').loc[g,'home_name']
        st.write(f'The {away_team} have a probability of {win_prob}% of beating the {home_team}.')

if __name__ == "__main__":
    main()
