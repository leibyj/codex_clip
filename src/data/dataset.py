import torch
# from torchvision.transforms import transforms
from torch.utils.data import Dataset
import pickle
import os


class CodexTextDataset(Dataset):
    def __init__(self, codex_data, tokenizer, max_len=256, **channel_groups):
        """
        Args:
            codex_data (dict): k: sample_id, v: tuple(codex_img, he_img, text_caption).
                               Includes the key "channel_biomarkers".
            tokenizer (transformers.PreTrainedTokenizer): HF tokenizer.
            max_len (int): context length for text data.
            channel_groups: Keyword args for marker groups. Example: 
                            tumor = ["C4d", "DAPI"], stroma = ['CD38', 'Tbet']
        """

        # CODEX variables
        # Extract channel biomarkers
        channel_biomarkers = list(codex_data["channel_biomarkers"])
        # Convert channel IDs to index (should add a check to make sure channel groups IDs are in channel_biomarkers)
        self.channel_inds = {}
        for k, v in channel_groups.items():
            self.channel_inds[k] = [channel_biomarkers.index(x) for x in v]

        # Text variables
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Exclude the "channel_biomarkers" key for getting sample_ids
        self.codex_data = {k: v for k, v in codex_data.items() if k != "channel_biomarkers"}
        self.sample_ids = list(self.codex_data.keys())

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        codex_img, caption = self.codex_data[sample_id]

        # for now, select biomarker_expression + cell_types sections
        caption = caption['biomarker_expression'] + " " + caption['cell_types'] 

        # CODEX processing
        codex_img = torch.tensor(codex_img, dtype=torch.float32)

        # create channel grouped images, including number of channels dict for ChannelViT
        grouped_imgs = {}
        for k, v in self.channel_inds.items():
            grouped_imgs[k] = (codex_img[v, :, :], {'channels': torch.tensor([len(v)])})

        # Text processing
        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        # return sample ID for debugging
        return {"codex": list(grouped_imgs.values()),
                "text": encoding['input_ids'].flatten(),
                "att_mask": encoding['attention_mask'].squeeze(0)}



class MultiCodexTextDataset(Dataset):
    def __init__(self, root_dir, region_ids, tokenizer, max_len=256, **channel_groups):
        """
        Args:
            root_dir (str): Directory where the .pkl files are stored.
            region_ids (list of str): List of region IDs to load.
            tokenizer (transformers.PreTrainedTokenizer): HF tokenizer.
            max_len (int): context length for text data.
            channel_groups: Keyword args for marker groups. Example: 
                            tumor = ["C4d", "DAPI"], stroma = ['CD38', 'Tbet']
        """
        self.root_dir = root_dir
        self.region_ids = region_ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.channel_groups = channel_groups
        
        # Load and store the data from all .pkl files
        self.files = {rid: self.load_pickle_file(rid) for rid in self.region_ids}

        # Flattened list of all (region_ids, sample_id) pairs
        self.sample_index = [
            (rid, sample_id) 
            for rid in self.region_ids 
            for sample_id in self.files[rid].keys()
        ]

        # Extract channel biomarkers from the first file
        channel_biomarkers = list(self.files[self.region_ids[0]]["channel_biomarkers"])
        
        # Convert channel IDs to indices
        self.channel_inds = {}
        for k, v in self.channel_groups.items():
            self.channel_inds[k] = [channel_biomarkers.index(x) for x in v]

    def load_pickle_file(self, region_id):
        """Load the pickle file corresponding to a patient."""
        file_path = os.path.join(self.root_dir, f'{region_id}.pkl')
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        patient_id, sample_id = self.sample_index[idx]
        codex_img, caption = self.files[patient_id][sample_id]

        # for now, select biomarker_expression + cell_types sections
        caption = caption['biomarker_expression'] + " " + caption['cell_types'] 

        # CODEX processing
        codex_img = torch.tensor(codex_img, dtype=torch.float32)

        # Create channel grouped images, including number of channels dict for ChannelViT
        grouped_imgs = {}
        for k, v in self.channel_inds.items():
            grouped_imgs[k] = (codex_img[v, :, :], {'channels': torch.tensor([len(v)])})

        # Text processing
        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        # Return the processed sample
        return {
            "codex": list(grouped_imgs.values()),
            "text": encoding['input_ids'].flatten(),
            "att_mask": encoding['attention_mask'].squeeze(0)
        }
