from dataclasses import dataclass



@dataclass
class ModelMetric:
    def __init__(self, model_name: str, metrics: dict):
        self.model_name = model_name
        self.metrics: dict[str, float] = metrics

    def get_metrics(self):
        return self.metrics

@dataclass
class ModelMetricsContainer:
    def __init__(self, model_metric: ModelMetric):
        self.model_metric_dict : dict[str, ModelMetric] = {}
        self.model_metric_dict[model_metric.model_name] = model_metric

    def __len__(self):
        return len(self.model_metric_dict)
    
    def append(self, model_metric: ModelMetric):
        self.model_metric_dict[model_metric.model_name] = model_metric

    def get_all(self):
        return self.model_metric_dict