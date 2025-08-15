import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional
import os

class CacheManager:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _generate_key(self, symbols: list, start_date: str, end_date: str) -> str:
        data_string = f"{'-'.join(sorted(symbols))}_{start_date}_{end_date}"
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def get_cached_data(self, symbols: list, start_date: str, end_date: str) -> Optional[Any]:
        key = self._generate_key(symbols, start_date, end_date)
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        
        if os.path.exists(cache_file):
            # Check if cache is still valid (e.g., less than 1 day old)
            if datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file)) < timedelta(days=1):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        return None
    
    def cache_data(self, data: Any, symbols: list, start_date: str, end_date: str):
        key = self._generate_key(symbols, start_date, end_date)
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)