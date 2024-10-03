import torch
# from torchvision.transforms import transforms
from torch.utils.data import Dataset
import pickle
import os
import torch.nn.functional as F

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
            region_dir = os.path.join(self.root_dir, rid)
            for sample_file in os.listdir(region_dir):
                if sample_file.endswith('.pkl'):
                    sample_id = os.path.splitext(sample_file)[0]
                    sample_index.append((rid, sample_id))
        return sample_index

    def load_sample(self, region_id, sample_id):
        """Load and return the sample corresponding to region_id and sample_id."""
        file_path = os.path.join(self.root_dir, region_id, f'{sample_id}.pkl')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            return data, data['channel_biomarkers']

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
    def __init__(self, root_dir, region_ids, tokenizer, max_len=256, transform=None):
        """
        Dataset class for loading full CODEX datasets without grouped channels.
        
        Args:
            root_dir (str): Directory where the .pkl files are stored.
            region_ids (list of str): List of region IDs to load.
            tokenizer (transformers.PreTrainedTokenizer): HF tokenizer
            max_len (int): context length for text data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root_dir, region_ids, tokenizer, max_len)
        self.transform = transform

    def __getitem__(self, idx):
        region_id, sample_id = self.sample_index[idx]
        dat, bms = self.load_sample(region_id, sample_id)
        
        codex_img = torch.tensor(dat['codex'], dtype=torch.float32)
        caption = dat['text']['biomarker_expression'] + " " + dat['text']['cell_types']

        if self.transform is not None:
            codex_img = torch.stack([self.transform(c.unsqueeze(0)).squeeze(0) for c in codex_img])

        input_ids, attention_mask = self.text_processing(caption)

        return {
            "codex": codex_img,
            "text": input_ids,
            "att_mask": attention_mask,
            "channels": bms,
            "region_id": region_id  # Add this line
        }

class MultiCodexDatasetFull(BaseCodexTextDataset):
    def __init__(self, root_dir, region_ids, tokenizer=None, max_len=256, transform=None):
        """
        Dataset class for loading full CODEX datasets without text captions.
        
        Args:
            root_dir (str): Directory where the .pkl files are stored.
            region_ids (list of str): List of region IDs to load.
            tokenizer (transformers.PreTrainedTokenizer): HF tokenizer
            max_len (int): context length for text data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root_dir, region_ids, tokenizer, max_len)
        self.transform = transform

    def __getitem__(self, idx):
        region_id, sample_id = self.sample_index[idx]
        dat, bms = self.load_sample(region_id, sample_id)
        
        codex_img = torch.tensor(dat['codex'], dtype=torch.float32)

        if self.transform is not None:
            codex_img = torch.stack([self.transform(c.unsqueeze(0)).squeeze(0) for c in codex_img])

        return {
            "codex": codex_img,
            "channels": bms,
            "region_id": region_id 
        }

class MultiCodexDatasetEval(BaseCodexTextDataset):
    def __init__(self, root_dir, region_ids, tokenizer=None, max_len=256, transform=None):
        """
        Args:
            root_dir (str): Directory where the .pkl files are stored.
            region_ids (list of str): List of region IDs to load.
            tokenizer (transformers.PreTrainedTokenizer): HF tokenizer
            max_len (int): context length for text data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__(root_dir, region_ids, tokenizer, max_len)
        self.transform = transform

    def build_sample_index(self):
        """Build and return a list of (region_id, sample_id) pairs."""
        sample_index = []
        for rid in self.region_ids:
            region_file = os.path.join(self.root_dir, f'{rid}_processed.pkl')
            with open(region_file, 'rb') as f:
                data = pickle.load(f)
                for sample_id in data.keys():
                    if sample_id == 'channel_biomarkers':
                        continue
                    sample_index.append((rid, sample_id))
        return sample_index

    def load_sample(self, region_id, sample_id):
        """Load and return the sample corresponding to region_id and sample_id."""
        region_file = os.path.join(self.root_dir, f'{region_id}_processed.pkl')
        with open(region_file, 'rb') as f:
            data = pickle.load(f)
            sample_data = data[sample_id]
            # print(sample_id)
            # print(sample_data.keys())
            return sample_data, data['channel_biomarkers']

    def __getitem__(self, idx):
        region_id, sample_id = self.sample_index[idx]
        dat, bms = self.load_sample(region_id, sample_id)
        
        codex_img = torch.tensor(dat['codex'], dtype=torch.float32)

        if self.transform is not None:
            codex_img = torch.stack([self.transform(c.unsqueeze(0)).squeeze(0) for c in codex_img])

        return {
            "codex": codex_img,
            "channels": bms,
            "region_id": region_id 
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
    # texts, attention_masks = None, None

    region_ids = [item['region_id'] for item in batch]  # Add this line

    return {
        "codex": padded_codex_imgs,
        "text": texts,
        "att_mask": attention_masks,
        "channels": channels,
        "region_id": region_ids  # Add this line
    }

def padded_collate_fn_codex_only(batch):

    # Find the maximum number of channels in the batch
    max_channels = max(item['codex'].shape[0] for item in batch)

    # Prepare lists for the padded codex images, texts, and attention masks
    padded_codex_imgs = []
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
        channels.append(item['channels'])

    # Stack tensors to create batch
    padded_codex_imgs = torch.stack(padded_codex_imgs)

    region_ids = [item['region_id'] for item in batch]  # Add this line

    return {
        "codex": padded_codex_imgs,
        "channels": channels,
        "region_id": region_ids  # Add this line
    }