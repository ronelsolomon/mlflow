#!/usr/bin/env python3
"""
Example usage of Kafka integration for real-time feature updates
"""

import threading
import time
from main import KafkaFeatureProducer, KafkaFeatureConsumer

def producer_example():
    """Example of producing real-time feature updates"""
    producer = KafkaFeatureProducer()
    
    # Simulate real-time data stream
    sample_data = [
        {"entity_id": "sensor_001", "temperature": 23.5, "humidity": 65.2, "pressure": 1013.25},
        {"entity_id": "sensor_002", "temperature": 24.1, "humidity": 63.8, "pressure": 1012.8},
        {"entity_id": "sensor_003", "temperature": 22.9, "humidity": 67.1, "pressure": 1013.5}
    ]
    
    print("Starting producer...")
    for i, data in enumerate(sample_data):
        entity_id = data.pop("entity_id")
        producer.publish_feature_update(entity_id, data)
        
        # Simulate prediction
        prediction = 0.8 if data["temperature"] > 23 else 0.3
        producer.publish_prediction(entity_id, prediction, data)
        
        print(f"Published update {i+1} for {entity_id}")
        time.sleep(2)
    
    producer.close()
    print("Producer finished")

def consumer_example():
    """Example of consuming real-time feature updates"""
    consumer = KafkaFeatureConsumer()
    
    print("Starting consumer (will run for 30 seconds)...")
    
    # Run consumer in a separate thread with timeout
    def run_consumer():
        start_time = time.time()
        while time.time() - start_time < 30:  # Run for 30 seconds
            consumer.consume_feature_updates(timeout_ms=2000)
    
    consumer_thread = threading.Thread(target=run_consumer)
    consumer_thread.start()
    consumer_thread.join()
    
    consumer.stop()
    print("Consumer finished")

if __name__ == "__main__":
    print("=== Kafka Real-time Feature Update Example ===")
    print("\n1. Run producer example")
    producer_example()
    
    print("\n2. Run consumer example")
    consumer_example()
