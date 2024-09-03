from tqdm import tqdm 
class Evaluator:
    def __init__(self):
        self.models_dict = None
        self.metrics_dict = None
        self.datasets_dict = None

        self.device = None
        self.current_model = None
        self.current_model_name = None
        self.current_dataset = None
        self.current_dataset_name = None
        
        self.labels = None
        self.all_logits = None

# LOADING FUNCTIONS
    def load_models_dict(self, models_dict):
        self.models_dict = models_dict
        return self.models_dict
           
    def set_device(self, device):
        self.device = device
        return self.device

    def load_datasets_dict(self, datasets_dict):
        self.datasets_dict = datasets_dict
        return self.datasets_dict
    
    def load_metrics_dict(self, metrics_dict):
        self.metrics_dict = metrics_dict
        return self.metrics_dict
    
    
# METRICS COMPUTATION
    def compute_metrics(self):
        computed_metrics = {}
        for _, metric in self.metrics_dict.items():
            result = metric.compute(predictions = self.all_logits, references = self.labels)
            computed_metrics.update(result)
        return computed_metrics


    def evaluate_qbq(self):
        self.current_model.eval()
        all_logits = []
        labels = []
        for index in tqdm(range(len(self.current_dataset)), desc=f"Evaluating {self.current_model_name} on {self.current_dataset_name}"):
            input_ids = self.current_dataset['input_ids'][index].unsqueeze(0).to(self.device)
            attention_mask = self.current_dataset['attention_mask'][index].unsqueeze(0).to(self.device)  
            token_type_ids = self.current_dataset['token_type_ids'][index].unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.current_model(input_ids = input_ids,
                                             attention_mask = attention_mask,
                                             token_type_ids = token_type_ids
                                            )
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)[0]
            label = self.current_dataset['labels'][index].to(self.device)
            
            all_logits.append(predictions)
            labels.append(label)
        self.all_logits = all_logits
        self.labels = labels
        
        evaluated_metrics = self.compute_metrics()
        return evaluated_metrics


    def evaluate_batch(self):
        self.current_model.eval()
        all_logits = []
        for batch in tqdm(self.current_dataset, desc="Evaluating"):
            inputs = {k:v.to(self.device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = self.current_model(**inputs)
            logits = outputs.logits
            all_logits.append(logits)
        self.all_logits = all_logits        
        evaluated_metrics = self.compute_metrics()
        return evaluated_metrics


    def evaluate_models(self):
        evaluation_results = {}
        for model_name, model in self.models_dict.items():
            self.current_model = model
            self.current_model_name = model_name
            result = self.evaluate_qbq()
            evaluation_results.update({model_name: result})
        return evaluation_results


    def evaluate_datasets(self):
        evaluation_results = {}
        for dataset_name, dataset in self.datasets_dict.items():
            self.current_dataset = dataset
            self.current_dataset_name = dataset_name
            result = self.evaluate_models()
            evaluation_results.update({dataset_name: result})
        return evaluation_results
    