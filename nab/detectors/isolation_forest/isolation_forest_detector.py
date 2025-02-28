from nab.detectors.base import AnomalyDetector
import numpy as np
from sklearn.ensemble import IsolationForest
from river import preprocessing  # For online scaling

class OnlineIsolationForestDetector(AnomalyDetector):
    def __init__(self, *args, **kwargs):
        super(OnlineIsolationForestDetector, self).__init__(*args, **kwargs)
        
        # Initialize parameters
        self.window_size = 100  # Sliding window size
        self.model = None
        self.scaler = preprocessing.MinMaxScaler()  # Online scaler
        self.buffer = []  # Stores recent data points
        self.training_data = []  # Buffers training data
        self.probationaryPeriod = 500  # Adjust based on NAB's requirement
        
    def handleRecord(self, inputData):
        """
        Processes one record at a time, returns anomaly score.
        """
        # Append new value to buffer
        self.buffer.append(inputData['value'])
        
        # 1. Buffer initialization phase
        if len(self.buffer) < self.window_size:
            return [0.0]  # Return neutral score during warm-up
        
        # 2. Extract current window
        current_window = self.buffer[-self.window_size:]
        
        # 3. Online scaling (using River's incremental scaler)
        scaled_window = [
            self.scaler.learn_one([x]).transform_one([x])[0] 
            for x in current_window
        ]
        
        # 4. Initial training during probationary period
        if self.record_count < self.probationaryPeriod:
            self.training_data.append(scaled_window)
            return [0.0]
        
        # 5. Initialize model after probationary period
        if self.model is None:
            self.model = IsolationForest(n_estimators=100, contamination=0.01)
            self.model.fit(np.array(self.training_data))
        
        # 6. Update model incrementally (partial fit)
        # Note: Isolation Forest doesn't support partial_fit, so we use sliding window retraining
        if self.record_count % self.window_size == 0:
            self.model.fit(np.array(self.training_data[-self.window_size:]))
        
        # 7. Compute anomaly score for current window
        anomaly_score = self.model.decision_function([scaled_window])[0]
        
        # 8. Convert to NAB-compatible score (higher = more anomalous)
        nab_score = -anomaly_score  # Isolation Forest outputs lower = more anomalous
        
        return [nab_score]