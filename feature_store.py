from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Any
import sqlite3
import os

class SimpleFeatureStore:
    """Simple feature store for ML pipeline integration"""
    
    def __init__(self, db_path: str = "feature_store.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database for feature storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT,
                feature_name TEXT,
                feature_value REAL,
                timestamp DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create feature metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_name TEXT PRIMARY KEY,
                description TEXT,
                data_type TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_features(self, entity_ids: List[str], feature_data: Dict[str, List], 
                       feature_descriptions: Dict[str, str] = None):
        """Store features for multiple entities"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now()
        
        # Store feature metadata
        if feature_descriptions:
            for feature_name, description in feature_descriptions.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO feature_metadata 
                    (feature_name, description, data_type)
                    VALUES (?, ?, ?)
                ''', (feature_name, description, 'numeric'))
        
        # Store feature values
        for i, entity_id in enumerate(entity_ids):
            for feature_name, values in feature_data.items():
                if i < len(values):
                    cursor.execute('''
                        INSERT INTO features 
                        (entity_id, feature_name, feature_value, timestamp)
                        VALUES (?, ?, ?, ?)
                    ''', (entity_id, feature_name, values[i], timestamp))
        
        conn.commit()
        conn.close()
    
    def get_features(self, entity_ids: List[str], feature_names: List[str]) -> pd.DataFrame:
        """Retrieve features for specific entities"""
        conn = sqlite3.connect(self.db_path)
        
        placeholders = ','.join(['?' for _ in entity_ids])
        feature_placeholders = ','.join(['?' for _ in feature_names])
        
        query = f'''
            SELECT entity_id, feature_name, feature_value, timestamp
            FROM features
            WHERE entity_id IN ({placeholders}) 
            AND feature_name IN ({feature_placeholders})
            ORDER BY entity_id, timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=entity_ids + feature_names)
        conn.close()
        
        # Pivot to get features as columns
        if not df.empty:
            pivot_df = df.pivot_table(
                index='entity_id', 
                columns='feature_name', 
                values='feature_value',
                aggfunc='first'
            ).reset_index()
            return pivot_df
        else:
            return pd.DataFrame()
    
    def get_feature_metadata(self) -> pd.DataFrame:
        """Get all feature metadata"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM feature_metadata', conn)
        conn.close()
        return df
    
    def list_features(self) -> List[str]:
        """List all available feature names"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT feature_name FROM features')
        features = [row[0] for row in cursor.fetchall()]
        conn.close()
        return features
