from tqdm import tqdm 
from evaluate import load
from torch.utils.data import DataLoader


class DataHelper:
    def __init__(self):
        self.datasets_dict = None        
        self.current_datasets_dict = {}
        self.formatted_datasets_dict = {}
        self.tokenized_datasets_dict = {}

        self.dataloaders_dict = {}

        self.tokenizer = None
#         self.tokenizer.pad_token = None

        self.system_instruction = "You are a Helpful AI Assistant."
        self.user_instruction = "Please answer the following Question: "
        self.user_query = None
        
        #datasets configurations
        self.batch_size = None
        self.shuffle = None
        self.max_length = None
        self.return_tensors = None
        self.padding = None
        self.truncation = None
        
        # Config Columns
        self.user_query_column = None        
        self.columns_to_tokenize = None  
        
        #Number
        self.number_one = 1
        self.number_two = 3
        

# DATASETS CLASS
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'input_ids': self.dataset['input_ids'][idx].unsqueeze(0),
            'attention_mask': self.dataset['attention_mask'][idx].unsqueeze(0),
            'token_type_ids': self.dataset.get('token_type_ids', torch.tensor([]))[idx].unsqueeze(0) if 'token_type_ids' in self.dataset else None
            }

# LOADING DATASETS DICT
    def load_datasets_dict(self, datasets_dict):
        self.datasets_dict = datasets_dict
        return self.datasets_dict

# LOADING DATASET CONFIGURATION
    def set_dataset_config(self, dataset_configuration):
        self.batch_size = dataset_configuration['batch_size']
        self.shuffle = dataset_configuration['shuffle']
        self.max_length = dataset_configuration['max_length']
        self.return_tensors = dataset_configuration['return_tensors']
        self.padding = dataset_configuration['padding']
        self.truncation = dataset_configuration['truncation']

# LOADING IMPORTANT COLUMNS
    def load_config_columns(self, columns_configuration):
        self.user_query_column = columns_configuration["user_query_column"]
        self.columns_to_tokenize = columns_configuration["columns_to_tokenize"]

# LOADING TOKENIZER
    def load_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        return self.tokenizer        

# SYSTEM & USER PROMPT
    def set_system_instruction(self, system_instruction):
        self.system_instruction = system_instruction
        return self.system_instruction
    
    def set_user_instruction(self, user_instruction):
        self.user_instruction = user_instruction
        return self.user_instruction
    
    def set_user_query(self, user_query):
        self.user_query = user_query
        return self.user_query

# HANDLING INPUT COLUMN
    def handle_dataset(self):
        pass

# CONVERTING DATASETS TO DATALOADER
    def datasets_to_dataloader(self):
        if self.tokenized_datasets_dict:
            self.current_datasets_dict = self.tokenized_datasets_dict
        elif self.formatted_datasets_dict:
            self.current_datasets_dict = self.formatted_datasets_dict
        else:
            self.current_datasets_dict = self.datasets_dict

        for dataset_name, dataset in self.current_datasets_dict.items():
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            self.dataloaders_dict.update({dataset_name+"_dataloader":dataloader})
        return self.dataloaders_dict
    
# FORMATTING DATASET CODE
    def convert_input_to_chat_template(self):
        message = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": self.user_instruction + self.user_query}
        ]
        formatted_input = self.tokenizer.apply_chat_template(message,
                                                                tokenize=False,
                                                                add_generation_prompt=True,
                                                                return_tensors = self.return_tensors
                                                            )
        return formatted_input

    def create_chat_template_dataset(self, example):
        self.user_query = example[self.user_query_column]
        example['training_input'] = self.convert_input_to_chat_template()
        return example

    def format_dataset(self, dataset):
        self.current_datasets_dict = self.datasets_dict
        
        for dataset_name, dataset in self.current_datasets_dict.items():
            formatted_dataset = dataset.map(self.create_chat_template_dataset)
            self.formatted_dataset_dict.update({dataset_name+"_formatted":formatted_dataset})
        return self.formatted_dataset_dict
    
# TOKENIZATION CODE    
    def tokenization_function(self, example):
        return self.tokenizer(example[self.columns_to_tokenize],
                                  padding=True,
                                  truncation=True,
                                  max_length=1024,
                                  return_tensors = self.return_tensors
                                 )
    
    def tokenize_datasets(self):
        if self.formatted_datasets_dict:
            self.current_datasets_dict = self.formatted_datasets_dict
        else:
            self.current_datasets_dict = self.datasets_dict
            
        for dataset_name, dataset in self.current_datasets_dict.items():
            tokenized_dataset = dataset.map(self.tokenization_function, batched=True, batch_size=128, num_proc=8)
            self.tokenized_datasets_dict.update({dataset_name+"_tokenized":dataset}) 
        return self.tokenized_datasets_dict    
    
    def remove_columns(self):
        dataset_name = list(self.datasets_dict.keys())[0]
        base_dataset_columns = list(self.datasets_dict[dataset_name].features.keys())
        tokenized_dataset_columns = list(self.current_datasets_dict[dataset_name].features.keys())
        final_columns = list(set(tokenized_dataset_columns) - set(base_dataset_columns))
        pass

