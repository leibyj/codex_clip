import torch
# from torchvision.transforms import transforms
from torch.utils.data import Dataset

class CodexTextDataset(Dataset):
    def __init__(self, codex_data, tokenizer, max_len=256, **channel_groups):
        """
        Args:
            codex_data (dict): k: sample_id, v: tuple(codex_img, he_img, text_caption).
                               Includes the key "channel_biomarkers".
            tokenizer (transformers.PreTrainedTokenizer): HF tokenizer
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