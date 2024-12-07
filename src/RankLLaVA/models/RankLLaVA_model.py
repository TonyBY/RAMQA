import torch
from torch import nn
from typing import List
from transformers import LlavaForConditionalGeneration, LlavaConfig
from transformers.modeling_outputs import ModelOutput
from transformers.activations import ACT2FN
from dataclasses import dataclass

from typing import Optional
import logging

logger = logging.getLogger()

    
class MLPClassificationHead(nn.Module):
        """Head for classification tasks."""
        def __init__(self, 
                     config: LlavaConfig, 
                     num_labels: int=2,
                     ):
            super().__init__()
            self.m = ACT2FN[config.projector_hidden_act]
            self.layer1  = nn.Linear(config.text_config.hidden_size, 512)
            self.layer2  = nn.Linear(512, 128)
            self.out_proj1 = nn.Linear(128, num_labels)

        def forward(self, x, **kwargs):
            # logger.info(f"x.shape: {x.shape}")
            x = self.layer1(x)
            x = self.m(x)
            x = self.layer2(x)
            x = self.m(x)
            x = self.out_proj1(x)
            return x
        
        
@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->Llava
class RankLLaVaOutput(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: Optional[torch.IntTensor] = None
    

class RankLLaVA(LlavaForConditionalGeneration):
    """
    Class for training LLaVa via Pytorch Lightning.
    """
    def __init__(
        self,
        back_bone_model=None,
        num_labels: int=None,
        class_weights: List[int]=None,
        label_smoothing: float=0.0,
    ):
        super().__init__(config=back_bone_model.config)

        self.model = back_bone_model
        self.config = back_bone_model.config
        self.classification_head = MLPClassificationHead(self.config)

        self.num_labels = num_labels

        self.class_weights = torch.FloatTensor(class_weights) if class_weights else torch.FloatTensor([1.0 for _ in range(num_labels)])
        logger.info(f"self.class_weights: {self.class_weights}")

        self.label_smoothing = label_smoothing

        self.classification_head = MLPClassificationHead(self.config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def forward(self, 
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                pixel_values: torch.FloatTensor = None,
                labels: Optional[torch.LongTensor] = None,
                normalize_to_unit: bool=True,
                is_test: bool=False,
                **kwargs,
                ):
        
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            output_hidden_states=True,
                            )
        # When passing output_hidden_states=True you may expect the outputs.hidden_states[-1] to match outputs.last_hidden_states exactly. 
        # However, this is not always the case. Some models apply normalization or subsequent process to the last hidden state when it’s returned.    
        last_hidden_state = output.hidden_states[-1] # BZ * num_tokens # emd_dim
        last_token_embeddings = last_hidden_state[:, -1] # BZ * emd_dim. 
        logits = self.classification_head(last_token_embeddings)

        if is_test:
             return RankLLaVaOutput(
                                loss=None,
                                logits=logits,
                                labels=None,
                                )
        
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing, 
                                             weight=self.class_weights.to(logits.device))
        
        logger.debug(f"labels.dtype: {labels.dtype}")
        loss = loss_fct(logits.view(-1, self.num_labels).to(torch.float32), labels.view(-1).to(logits.device))

        return RankLLaVaOutput(
                                loss=loss,
                                logits=logits,
                                labels=labels,
                                )
