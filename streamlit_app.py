import streamlit as st
import os
import itertools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta,time as dt_time
from sklearn.model_selection import train_test_split
import glob
import random
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)

class LinearRegression(object):
    def __init__(self, learning_rate=0.001, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.cost_history = []
        self.W = None

    def transform(self, X):
        return X

    def predict(self, X):
        X = np.array(X)

        X_transformed = self.transform(X)

        if hasattr(self, 'mean') and hasattr(self, 'std'):
            X_normalized = (X_transformed - self.mean) / self.std
        else:
            X_normalized = X_transformed

        X_with_ones = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]
        W = self.W
        if W.ndim == 1:
            W = W.reshape(-1, 1)

        predictions = X_with_ones.dot(W)

        return predictions.flatten()

    def update_weights(self):
        if self.W.ndim == 1:
            self.W = self.W.reshape(-1, 1)

        Y_pred = self.X_with_ones.dot(self.W)

        error = Y_pred - self.Y

        gradient = (2 * self.X_with_ones.T.dot(error)) / self.num_examples
        self.W -= self.learning_rate * gradient

        self.cost_history.append(np.mean((error) ** 2))

        return self

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        self.original_n_features = X.shape[1]

        X_transformed = self.transform(X)

        self.mean = np.mean(X_transformed, axis=0)
        self.std = np.std(X_transformed, axis=0)
        self.std[self.std == 0] = 1
        X_normalized = (X_transformed - self.mean) / self.std

        self.X_with_ones = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]

        self.W = np.zeros((self.X_with_ones.shape[1], 1))

        self.X = X_normalized
        self.Y = Y.reshape(-1, 1)
        self.num_examples = X_normalized.shape[0]

        for i in range(self.num_iterations):
            self.update_weights()

            if i > 1 and abs(self.cost_history[-1] - self.cost_history[-2]) < 1e-6:
                break

        return self

    def get_r2(self, X, Y):
        Y_pred = self.predict(X)
        Y = np.array(Y)

        SS_res = np.sum((Y - Y_pred) ** 2)
        SS_tot = np.sum((Y - np.mean(Y)) ** 2)
        r2 = 1 - SS_res / SS_tot

        return r2

    def get_mse(self, X, Y):
        Y_pred = self.predict(X)
        Y = np.array(Y)

        return np.mean((Y - Y_pred) ** 2)

    def get_mae(self, X, Y):
        y_pred = self.predict(X)
        y_true = np.array(Y)
        num_examples = len(y_true)
        error = (np.sum(np.abs(y_pred - y_true))) / num_examples
        return error


class PolynomialRegression(LinearRegression):
    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        self.degree = degree
        super().__init__(learning_rate, num_iterations)

    def transform(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        num_examples, num_features = X.shape
        features = []

        for j in range(0, self.degree + 1):
            if j == 0:
                continue
            for combinations in itertools.combinations_with_replacement(range(num_features), j):
                feature = np.ones(num_examples)
                for each_combination in combinations:
                    feature = feature * X[:, each_combination]
                features.append(feature.reshape(-1, 1))

        if features:
            X_transform = np.concatenate(features, axis=1)
        else:
            X_transform = X

        return X_transform


class KNN(object):
    def __init__(self, k=20, metric='Minkowski', p=1):
        self.model_name = 'K Nearest Neighbor Regressor'
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None
        self.is_fitted = False

        self.mean = None
        self.std = None

    def euclidean_distance(self, X, query):
        try:
            distances = []
            for q in query:
                dist = np.sqrt(np.sum((X - q) ** 2, axis=1))
                distances.append(dist)
            return np.array(distances)
        except ValueError as err:
            print(f"Error in euclidean_distance: {str(err)}")
            return None

    def manhattan_distance(self, X, query):
        try:
            distances = []
            for q in query:
                dist = np.sum(np.abs(X - q), axis=1)
                distances.append(dist)
            return np.array(distances)
        except ValueError as err:
            print(f"Error in manhattan_distance: {str(err)}")
            return None

    def minkowski_distance(self, X, query):
        try:
            distances = []
            for q in query:
                dist = np.power(np.sum(np.power(np.abs(X - q), self.p), axis=1), 1 / self.p)
                distances.append(dist)
            return np.array(distances)
        except ValueError as err:
            print(f"Error in minkowski_distance: {str(err)}")
            return None

    def calculate_distance(self, X, query):
        if self.metric == 'Euclidean':
            return self.euclidean_distance(X, query)
        elif self.metric == 'Manhattan':
            return self.manhattan_distance(X, query)
        elif self.metric == 'Minkowski':
            return self.minkowski_distance(X, query)
        else:
            raise ValueError(f"Not Supported metrics: {self.metric}")

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        self.mean = np.mean(self.X_train, axis=0)
        self.std = np.std(self.X_train, axis=0)
        self.std[self.std == 0] = 1

        self.X_train_normalized = (self.X_train - self.mean) / self.std
        self.is_fitted = True
        return self

    def kneighbors(self, X, k=None):
        if k is None:
            k = self.k

        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X_normalized = (X - self.mean) / self.std

        distances = self.calculate_distance(self.X_train_normalized, X_normalized)

        all_indices = []
        all_distances = []

        for i, dist in enumerate(distances):
            sorted_indices = np.argsort(dist)[:k]
            all_indices.append(sorted_indices)
            all_distances.append(dist[sorted_indices])

        return np.array(all_distances), np.array(all_indices)

    def predict(self, X):
        X = np.array(X)
        _, neighbor_indices = self.kneighbors(X)

        predictions = []
        for indices in neighbor_indices:
            predictions.append(np.mean(self.y_train[indices]))

        return np.array(predictions)

    def get_r2(self, X, y):
        y_pred = self.predict(X)
        y = np.array(y)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)

    def get_mse(self, X, Y):
        Y_pred = self.predict(X)
        Y = np.array(Y)
        return np.mean((Y - Y_pred) ** 2)

    def get_mae(self, X, Y):
        y_pred = self.predict(X)
        y_true = np.array(Y)
        num_examples = len(y_true)
        error = (np.sum(np.abs(y_pred - y_true))) / num_examples
        return error

def preprocess_data(df):
    print("preprocess data...")

    df['started_at'] = pd.to_datetime(df['started_at'], format='mixed', errors='coerce')
    df['ended_at'] = pd.to_datetime(df['ended_at'], format='mixed', errors='coerce')

    df = df.dropna(subset=['start_station_id', 'end_station_id', 'started_at', 'ended_at'])

    df['ride_duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

    df = df[(df['ride_duration_min'] > 0) & (df['ride_duration_min'] < 1440)]

    return df

def generate_station_features(df, station_id):
    station_starts = df[df['start_station_id'] == station_id]
    station_ends = df[df['end_station_id'] == station_id]

    if station_starts.empty and station_ends.empty:
        return pd.DataFrame()

    if 'start_station_name' in df.columns and not station_starts.empty:
        station_name = station_starts['start_station_name'].iloc[0]
    elif 'end_station_name' in df.columns and not station_ends.empty:
        station_name = station_ends['end_station_name'].iloc[0]
    else:
        station_name = station_id

    start_date = pd.to_datetime("2024-01-01")
    end_date = pd.to_datetime("2024-12-31 23:00")
    hours = pd.date_range(start=start_date, end=end_date, freq='h')

    flow_data = []
    for hour in hours:
        hour_end = hour + timedelta(hours=1)

        out_flow = station_starts[(station_starts['started_at'] >= hour) &
                                  (station_starts['started_at'] < hour_end)].shape[0]

        in_flow = station_ends[(station_ends['ended_at'] >= hour) &
                               (station_ends['ended_at'] < hour_end)].shape[0]

        total_flow = in_flow + out_flow

        flow_data.append({
            'datetime': hour,
            'station_id': station_id,
            'station_name': station_name,
            'total_flow': total_flow,
            'hour': hour.hour,
            'day_of_week': hour.dayofweek,
            'is_weekend': 1 if hour.dayofweek >= 5 else 0,
        })

    flow_df = pd.DataFrame(flow_data)

    flow_df = flow_df.sort_values('datetime')
    flow_df['previous_1h_flow'] = flow_df['total_flow'].shift(1).fillna(0)

    past_24h_flows = []
    for i in range(len(flow_df)):
        past_flows = flow_df['total_flow'].iloc[max(0, i - 24):i]
        avg_flow = past_flows.mean() if len(past_flows) > 0 else 0
        past_24h_flows.append(avg_flow)
    flow_df['past_day_avg_flow'] = past_24h_flows

    flow_df['previous_week_flow'] = flow_df['total_flow'].shift(24 * 7).fillna(0)

    return flow_df

def make_future_features(station_df, dt):
    hour = dt.hour
    dow  = dt.weekday()
    is_weekend = 1 if dow >= 5 else 0

    hist = station_df

    avg_hour_flow = hist[hist['hour']==hour]['total_flow'].mean()

    window = hist[
        (hist['datetime'] >= dt - timedelta(days=1)) &
        (hist['datetime'] <  dt)
    ]
    past_day_avg = window['total_flow'].mean() if not window.empty else avg_hour_flow

    week_ago = hist[hist['datetime'] == dt - timedelta(weeks=1)]
    prev_week_flow = week_ago['total_flow'].iloc[0] if not week_ago.empty else avg_hour_flow

    return np.array([[hour, dow, is_weekend, avg_hour_flow, past_day_avg, prev_week_flow]])

@st.cache_data(show_spinner=True)
def load_and_filter_station(data_folder, station_id):
    df_chunks = []
    for f in glob.glob(os.path.join(data_folder, "*.csv")):
        for chunk in pd.read_csv(f,
                                  usecols=['start_station_id','end_station_id','started_at','ended_at'],
                                  parse_dates=['started_at','ended_at'],
                                  chunksize=200_000):
            mask = (chunk['start_station_id']==station_id) | (chunk['end_station_id']==station_id)
            if mask.any():
                df_chunks.append(chunk[mask])
    if not df_chunks:
        return pd.DataFrame()
    df = pd.concat(df_chunks, ignore_index=True)
    return preprocess_data(df)

@st.cache_data(show_spinner=True)
def get_station_data(df: pd.DataFrame, station_id: str) -> pd.DataFrame:
    return generate_station_features(df, station_id)

@st.cache_resource(show_spinner=True)
def train_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray):
    if model_name == 'Linear Regression':
        model = LinearRegression(learning_rate=0.001, num_iterations=3000)
    elif model_name == 'Polynomial Regression (degree=2)':
        model = PolynomialRegression(degree=2, learning_rate=0.001, num_iterations=3000)
    elif model_name == 'Polynomial Regression (degree=3)':
        model = PolynomialRegression(degree=3, learning_rate=0.001, num_iterations=3000)
    elif model_name == 'KNN (k=5)':
        model = KNN(k=5, metric='Euclidean')
    elif model_name == 'KNN (k=10)':
        model = KNN(k=10, metric='Euclidean')
    elif model_name == 'KNN (k=20)':
        model = KNN(k=20, metric='Euclidean')
    elif model_name == 'KNN Manhattan (k=10)':
        model = KNN(k=10, metric='Manhattan')
    elif model_name == 'KNN Minkowski (k=10, p=2)':
        model = KNN(k=10, metric='Minkowski', p=2)
    else:
        st.error(f"unknow model: {model_name}")
        return None
    model.fit(X_train, y_train)
    return model


st.set_page_config(page_title="Citibike Flow Prediction", layout="wide")
st.title("Citibike flow prediction system")

st.sidebar.header("setting")

data_folder = st.sidebar.text_input("file path for datasets", value=os.path.expanduser("data"))
station_id = st.sidebar.text_input("Station ID", value="")
model_name = st.sidebar.selectbox(
    "Choose prediction model", [
        'Linear Regression',
        'Polynomial Regression (degree=2)',
        'Polynomial Regression (degree=3)',
        'KNN (k=5)',
        'KNN (k=10)',
        'KNN (k=20)',
        'KNN Manhattan (k=10)',
        'KNN Minkowski (k=10, p=2)'
    ]
)

selected_date = st.sidebar.date_input("Choose date", value=datetime.now().date())
selected_time = st.sidebar.time_input("Choose time", value=dt_time(datetime.now().hour, 0))
split_ratio = st.sidebar.slider("train/test split ratio", min_value=0.1, max_value=0.9, value=0.8, step=0.05)

if st.sidebar.button("start prediction"):
    df = load_and_filter_station(data_folder, station_id)
    station_df = generate_station_features(df, station_id)

    if station_df.empty:
        st.error("No data found in this station, please check the station id。")
    else:

        features = ['hour', 'day_of_week', 'is_weekend', 'previous_1h_flow', 'past_day_avg_flow', 'previous_week_flow']
        target = 'total_flow'

        X = station_df[features]
        y = station_df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=1 - split_ratio,random_state=42)

        model = train_model(model_name, X_train.values, y_train.values)
        if model:
            y_pred_test = model.predict(X_test.values)
            mse = model.get_mse(X_test.values, y_test.values)
            mae = model.get_mae(X_test.values, y_test.values)
            r2 = model.get_r2(X_test.values, y_test.values)

            st.subheader("Model evaluation metrics（test set）")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"RMSE: {np.sqrt(mse):.2f}")
            st.write(f"R²: {r2:.2f}")

            dt = datetime.combine(selected_date, selected_time).replace(minute=0, second=0, microsecond=0)
            if dt <= station_df['datetime'].max():
                row = station_df[station_df['datetime'] == dt]
                X_pred = row[features].values
            else:
                X_pred = make_future_features(station_df, dt)
            pred_flow = model.predict(X_pred)[0]

            st.subheader(f"predict total flow: **{int(pred_flow)} times**")

            test_idx = X_test.index
            plot_df = station_df.loc[test_idx].copy()

            plot_df['predicted_flow'] = y_pred_test

            plot_df = plot_df.set_index('datetime')[['total_flow', 'predicted_flow']]

            st.subheader("Timing diagram: True vs Predicted（test set）")
            st.line_chart(plot_df)