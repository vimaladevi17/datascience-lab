"""
Experiment Title: Simulation of Scalable Data Pipeline
Stages: Ingestion -> Processing -> Output
Technology: Python (Threading, Queue, Pandas)
"""

import threading
import queue
import random
import time
import pandas as pd

# Create shared queues
ingestion_queue = queue.Queue()
processing_queue = queue.Queue()

# ----------- STAGE 1: DATA INGESTION -----------
def data_ingestion():
    """
    Simulates streaming data ingestion from sensors
    """
    for i in range(100):   # simulate 100 records
        data = {
            "sensor_id": random.randint(1, 5),
            "temperature": random.uniform(20, 40),
            "humidity": random.uniform(30, 90),
            "timestamp": time.time()
        }
        ingestion_queue.put(data)
        print(f"Ingested: {data}")
        time.sleep(0.05)   # simulate real-time delay
    
    ingestion_queue.put(None)  # signal completion


# ----------- STAGE 2: DATA PROCESSING -----------
def data_processing():
    """
    Cleans and transforms incoming data
    """
    while True:
        data = ingestion_queue.get()
        
        if data is None:
            processing_queue.put(None)
            break
        
        # Data cleaning
        if data["temperature"] > 25:
            data["status"] = "HIGH"
        else:
            data["status"] = "NORMAL"
        
        processing_queue.put(data)
        print(f"Processed: {data}")


# ----------- STAGE 3: DATA OUTPUT -----------
def data_output():
    """
    Stores processed data into CSV file
    """
    processed_data = []
    
    while True:
        data = processing_queue.get()
        
        if data is None:
            break
        
        processed_data.append(data)
        print(f"Output Ready: {data}")
    
    # Save to CSV
    df = pd.DataFrame(processed_data)
    df.to_csv("processed_sensor_data.csv", index=False)
    print("\nData successfully saved to processed_sensor_data.csv")


# ----------- MAIN EXECUTION -----------
if __name__ == "__main__":
    t1 = threading.Thread(target=data_ingestion)
    t2 = threading.Thread(target=data_processing)
    t3 = threading.Thread(target=data_output)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print("\nPipeline Execution Completed Successfully!")