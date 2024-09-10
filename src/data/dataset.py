import torch
# from torchvision.transforms import transforms
from torch.utils.data import Dataset
import pickle
import os
import torch.nn.functional as F


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



# class MultiCodexTextDataset(Dataset):
#     def __init__(self, root_dir, region_ids, tokenizer, max_len=256, **channel_groups):
#         """
#         Args:
#             root_dir (str): Directory where the .pkl files are stored.
#             region_ids (list of str): List of region IDs to load.
#             tokenizer (transformers.PreTrainedTokenizer): HF tokenizer
#             max_len (int): context length for text data.
#             channel_groups: Keyword args for marker groups. Example: 
#                             tumor = ["C4d", "DAPI"], stroma = ['CD38', 'Tbet']
#         """
#         self.root_dir = root_dir
#         self.region_ids = region_ids
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.channel_groups = channel_groups
        
#         # Store a list of all (region_id, sample_id) pairs
#         self.sample_index = self.build_sample_index()

#     def build_sample_index(self):
#         """Build and return a list of (region_id, sample_id) pairs."""
#         sample_index = []
#         for rid in self.region_ids:
#             file_path = os.path.join(self.root_dir, f'{rid}_processed_subset.pkl')
#             with open(file_path, 'rb') as f:
#                 data = pickle.load(f)
#                 for sample_id in data.keys():
#                     if sample_id != "channel_biomarkers":
#                         sample_index.append((rid, sample_id))
#         return sample_index

#     def load_sample(self, region_id, sample_id):
#         """Load and return the sample corresponding to region_id and sample_id."""
#         file_path = os.path.join(self.root_dir, f'{region_id}_processed_subset.pkl')
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
#             return data[sample_id], data['channel_biomarkers']

#     def __len__(self):
#         return len(self.sample_index)

#     def __getitem__(self, idx):
#         region_id, sample_id = self.sample_index[idx]
#         dat, bms = self.load_sample(region_id, sample_id)
#         # codex_img, caption = dat
#         codex_img = dat['codex']
#         caption = dat['text']
#         # For now, select biomarker_expression + cell_types sections
#         caption = caption['biomarker_expression'] + " " + caption['cell_types'] 

#         # CODEX processing
#         codex_img = torch.tensor(codex_img, dtype=torch.float32)

#         # Create channel grouped images, including number of channels dict for ChannelViT
#         # get channel index (could be unique across studies)
#         channel_inds = {}
#         for k, v in self.channel_groups.items():
#             channel_inds[k] = [bms.index(x) for x in v]
#         grouped_imgs = {}
#         for k, v in channel_inds.items():
#             grouped_imgs[k] = (codex_img[v, :, :], {'channels': torch.tensor([len(v)])})

#         # Text processing
#         encoding = self.tokenizer.encode_plus(
#             caption,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding='max_length',
#             return_attention_mask=True,
#             return_tensors='pt',
#             truncation=True
#         )

        
#         return {
#             "codex": list(grouped_imgs.values()),
#             "text": encoding['input_ids'].flatten(),
#             "att_mask": encoding['attention_mask'].squeeze(0)
#         }


# class MultiCodexTextDataset_2(Dataset):
#     def __init__(self, root_dir, region_ids, tokenizer, max_len=256, **channel_groups):
#         """
#         Args:
#             root_dir (str): Directory where the .pkl files are stored.
#             region_ids (list of str): List of region IDs to load.
#             tokenizer (transformers.PreTrainedTokenizer): HF tokenizer
#             max_len (int): context length for text data.
#             channel_groups: Keyword args for marker groups. Example: 
#                             tumor = ["C4d", "DAPI"], stroma = ['CD38', 'Tbet']
#         """
#         self.root_dir = root_dir
#         self.region_ids = region_ids
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.channel_groups = channel_groups
        
#         # Store a list of all (region_id, sample_id) pairs
#         self.sample_index = self.build_sample_index()

#     def build_sample_index(self):
#         """Build and return a list of (region_id, sample_id) pairs."""
#         sample_index = []
#         for rid in self.region_ids:
#             file_path = os.path.join(self.root_dir, f'{rid}_processed_subset.pkl')
#             with open(file_path, 'rb') as f:
#                 data = pickle.load(f)
#                 for sample_id in data.keys():
#                     if sample_id != "channel_biomarkers":
#                         sample_index.append((rid, sample_id))
#         return sample_index

#     def load_sample(self, region_id, sample_id):
#         """Load and return the sample corresponding to region_id and sample_id.
#            .pkl files consist of dictionary with keys = patch_ids + 'channel_biomarkers'
#         """
#         file_path = os.path.join(self.root_dir, f'{region_id}_processed_subset.pkl')
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
#             return data[sample_id], data['channel_biomarkers'] # sample_id is patch name

#     def __len__(self):
#         return len(self.sample_index)

#     def __getitem__(self, idx):
#         region_id, sample_id = self.sample_index[idx]
#         dat, bms = self.load_sample(region_id, sample_id)
#         # codex_img, caption = dat
#         codex_img = dat['codex']
#         caption = dat['text']
#         # For now, select biomarker_expression + cell_types sections
#         caption = caption['biomarker_expression'] + " " + caption['cell_types'] 

#         # CODEX processing
#         codex_img = torch.tensor(codex_img, dtype=torch.float32)

#         # Text processing
#         encoding = self.tokenizer.encode_plus(
#             caption,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding='max_length',
#             return_attention_mask=True,
#             return_tensors='pt',
#             truncation=True
#         )

        
#         return {
#             "codex": codex_img,
#             "text": encoding['input_ids'].flatten(),
#             "att_mask": encoding['attention_mask'].squeeze(0),
#             "channels": bms
#         }

import os
import pickle
import torch
from torch.utils.data import Dataset

class BaseCodexTextDataset(Dataset):
    def __init__(self, root_dir, region_ids, tokenizer, max_len=256):
        """
        Base class for loading CODEX datasets.
        
        Args:
            root_dir (str): Directory where the .pkl files are stored.
            region_ids (list of str): List of region IDs to load.
            tokenizer (transformers.PreTrainedTokenizer): HF tokenizer
            max_len (int): context length for text data.
        """
        self.root_dir = root_dir
        self.region_ids = region_ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Store a list of all (region_id, sample_id) pairs
        self.sample_index = self.build_sample_index()

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

    def text_processing(self, caption):
        """Process the text data."""
        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return encoding['input_ids'].flatten(), encoding['attention_mask'].squeeze(0)

    def __len__(self):
        return len(self.sample_index)

class MultiCodexTextDatasetSubset(BaseCodexTextDataset):
    def __init__(self, root_dir, region_ids, tokenizer, max_len=256, **channel_groups):
        """
        Dataset with grouped channel images.
        
        Args:
            root_dir (str): Directory where the .pkl files are stored.
            region_ids (list of str): List of region IDs to load.
            tokenizer (transformers.PreTrainedTokenizer): HF tokenizer
            max_len (int): context length for text data.
            channel_groups: Keyword args for marker groups. Example: 
                            tumor = ["C4d", "DAPI"], stroma = ['CD38', 'Tbet']
        """
        super().__init__(root_dir, region_ids, tokenizer, max_len)
        self.channel_groups = channel_groups

    def __getitem__(self, idx):
        region_id, sample_id = self.sample_index[idx]
        dat, bms = self.load_sample(region_id, sample_id)
        
        codex_img = torch.tensor(dat['codex'], dtype=torch.float32)
        caption = dat['text']['biomarker_expression'] + " " + dat['text']['cell_types']
        
        # Create channel grouped images
        channel_inds = {k: [bms.index(x) for x in v] for k, v in self.channel_groups.items()}
        grouped_imgs = {k: (codex_img[v, :, :], {'channels': torch.tensor([len(v)])}) for k, v in channel_inds.items()}

        # Text processing
        input_ids, attention_mask = self.text_processing(caption)

        return {
            "codex": list(grouped_imgs.values()),
            "text": input_ids,
            "att_mask": attention_mask
        }

class MultiCodexTextDatasetFull(BaseCodexTextDataset):
    def __init__(self, root_dir, region_ids, tokenizer, max_len=256):
        """
        Dataset class for loading full CODEX datasets without grouped channels.
        
        Args:
            root_dir (str): Directory where the .pkl files are stored.
            region_ids (list of str): List of region IDs to load.
            tokenizer (transformers.PreTrainedTokenizer): HF tokenizer
            max_len (int): context length for text data.
        """
        super().__init__(root_dir, region_ids, tokenizer, max_len)

    def __getitem__(self, idx):
        region_id, sample_id = self.sample_index[idx]
        dat, bms = self.load_sample(region_id, sample_id)
        
        codex_img = torch.tensor(dat['codex'], dtype=torch.float32)
        caption = dat['text']['biomarker_expression'] + " " + dat['text']['cell_types']

        # Text processing
        input_ids, attention_mask = self.text_processing(caption)

        return {
            "codex": codex_img,
            "text": input_ids,
            "att_mask": attention_mask,
            "channels": bms
        }

def padded_collate_fn(batch):

    # Find the maximum number of channels in the batch
    max_channels = max(item['codex'].shape[0] for item in batch)

    # Prepare lists for the padded codex images, texts, and attention masks
    padded_codex_imgs = []
    texts = []
    attention_masks = []
    channels = []

    for item in batch:
        codex_img = item['codex']
        c, h, w = codex_img.shape

        # Pad the codex image to match the max number of channels in the batch
        if c < max_channels:
            padding = (0, 0, 0, 0, 0, max_channels - c)  # Pad only on the channel dimension
            padded_img = F.pad(codex_img, padding, mode='constant', value=0)
        else:
            padded_img = codex_img

        padded_codex_imgs.append(padded_img)
        texts.append(item['text'])
        attention_masks.append(item['att_mask'])
        channels.append(item['channels'])

    # Stack tensors to create batch
    padded_codex_imgs = torch.stack(padded_codex_imgs)
    texts = torch.stack(texts)
    attention_masks = torch.stack(attention_masks)

    return {
        "codex": padded_codex_imgs,
        "text": texts,
        "att_mask": attention_masks,
        "channels": channels
    }