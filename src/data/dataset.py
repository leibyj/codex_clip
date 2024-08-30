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
            tokenizer (transformers.PreTrainedTokenizer): HF tokenizer
            max_len (int): context length for text data.
            channel_groups: Keyword args for marker groups. Example: 
                            tumor = ["C4d", "DAPI"], stroma = ['CD38', 'Tbet']
        """
        self.root_dir = root_dir
        self.region_ids = region_ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.channel_groups = channel_groups
        
        # Store a list of all (region_id, sample_id) pairs
        self.sample_index = self.build_sample_index()

        ### THIS ASSUMES ALL MARKERS ARE IN SAME ORDER AS IN FIRST FILE
        # Extract channel biomarkers from the first file
        # with open(os.path.join(self.root_dir, f'{self.region_ids[0]}_processed.pkl'), 'rb') as f:
        #     file = pickle.load(f)
        #     channel_biomarkers = list(file["channel_biomarkers"])
        
        # # Convert channel IDs to indices
        # self.channel_inds = {}
        # for k, v in self.channel_groups.items():
        #     self.channel_inds[k] = [channel_biomarkers.index(x) for x in v]

    def build_sample_index(self):
        """Build and return a list of (region_id, sample_id) pairs."""
        sample_index = []
        for rid in self.region_ids:
            file_path = os.path.join(self.root_dir, f'{rid}_processed.pkl')
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                for sample_id in data.keys():
                    if sample_id != "channel_biomarkers":
                        sample_index.append((rid, sample_id))
        return sample_index

    def load_sample(self, region_id, sample_id):
        """Load and return the sample corresponding to region_id and sample_id."""
        file_path = os.path.join(self.root_dir, f'{region_id}_processed.pkl')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            return data[sample_id], data['channel_biomarkers']

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        region_id, sample_id = self.sample_index[idx]
        codex_img, caption, bms = self.load_sample(region_id, sample_id)

        # For now, select biomarker_expression + cell_types sections
        caption = caption['biomarker_expression'] + " " + caption['cell_types'] 

        # CODEX processing
        codex_img = torch.tensor(codex_img, dtype=torch.float32)

        # Create channel grouped images, including number of channels dict for ChannelViT
        # get channel index (could be unique across studies)
        for k, v in self.channel_groups.items():
            channel_inds[k] = [bms.index(x) for x in v]
        grouped_imgs = {}
        for k, v in channel_inds.items():
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

        
        return {
            "codex": list(grouped_imgs.values()),
            "text": encoding['input_ids'].flatten(),
            "att_mask": encoding['attention_mask'].squeeze(0)
        }
