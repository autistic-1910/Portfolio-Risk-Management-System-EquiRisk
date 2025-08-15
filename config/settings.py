from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class AnalysisConfig:
    confidence_levels: List[float] = None
    monte_carlo_iterations: int = 10000
    var_methods: List[str] = None
    stress_test_scenarios: Dict = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]
        if self.var_methods is None:
            self.var_methods = ['historical', 'monte_carlo', 'garch']
        if self.stress_test_scenarios is None:
            self.stress_test_scenarios = {
                'covid_2020': {'start': '2020-02-01', 'end': '2020-04-01'},
                'financial_crisis_2008': {'start': '2008-09-01', 'end': '2009-03-01'}
            }
    
    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)