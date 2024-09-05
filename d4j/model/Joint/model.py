import torch.nn as nn
import torch
import numpy as np
import os
from transformers import (T5ForConditionalGeneration, RobertaTokenizer)
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.utils.doc import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from typing import Optional, Tuple, Union
import logging


class DataBase(object):
    def __init__(self, vector) -> None:
        self.vector = vector
        self.history = []

    def __len__(self):
        return self.vector.shape[0]

    def search(self, query, number, stage=None):
        scores = np.matmul(query, self.vector.T)
        sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

        index = []
        for i in range(len(sort_ids)):
            if stage == "train":
                # 对于 train 来说要排除自己
                index.append(sort_ids[i][1:number + 1])
            else:
                index.append(sort_ids[i][0:number])

        if stage == "train":
            self.history.append(index)

        return index

    def get_history(self):
        temp = self.history
        self.history = []
        return temp

    def update(self, index, vectors):
        for id, vector in zip(index, vectors):
            self.vector[id] = vector


logger = logging.getLogger(__name__)

MODEL_CLASSES = (T5Config, T5ForConditionalGeneration, RobertaTokenizer)


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def build_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    # load generator
    generator = Generator.from_pretrained(args.model_name_or_path)
    if hasattr(args, 'beam_size'):
        beam_size = args.beam_size
    else:
        beam_size = 10
    if hasattr(args, 'max_target_length'):
        max_target_length = args.max_target_length
    else:
        max_target_length = 64

    generator._set(tokenizer, beam_size, max_target_length)
    if args.load_generator_from_checkpoint:
        checkpoint_prefix = f'checkpoint-best-bleu/{args.checkpoint_retriever_name}'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        generator.load_state_dict(torch.load(output_dir))

    # load retriever
    retriever = Retriever.from_pretrained(args.model_name_or_path)
    retriever._set(tokenizer)
    if args.load_retriever_from_checkpoint:
        checkpoint_prefix = f'checkpoint-best-bleu/{args.checkpoint_retriever_name}'
        output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}")
        retriever.load_state_dict(torch.load(output_dir))

    logger.info("Finish loading generator [%s] from %s", get_model_size(generator), args.model_name_or_path)
    logger.info("Finish loading retriever [%s] from %s", get_model_size(retriever), args.model_name_or_path)

    return config, generator, retriever, tokenizer


class Generator(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

    def _set(self, tokenizer: RobertaTokenizer, beam_size: int, max_length: int):
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_length = max_length

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            score=None,
            is_generate=False,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""

        if is_generate:
            preds = self.generate(input_ids,
                                  attention_mask=attention_mask,
                                  use_cache=True,
                                  num_beams=self.beam_size,
                                  early_stopping=True,
                                  max_length=self.max_length)
            if preds.size(1) < self.max_length:
                padding = torch.zeros((preds.size(0), self.max_length - preds.size(1)), dtype=preds.dtype).cuda()
                preds = torch.cat([preds, padding], dim=1)
            return preds

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = None
            for idx, (logit, label) in enumerate(zip(lm_logits, labels)):
                _loss = loss_fct(logit.squeeze(), label.squeeze())
                if score is not None:
                    _loss = _loss * score[idx]

                if loss is None:
                    loss = _loss
                else:
                    loss += _loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            logger.info(
                f"lm_logits type: {type(lm_logits)}, decoder_outputs type: {type(decoder_outputs)}, encoder_outputs type: {type(encoder_outputs)}")
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class Retriever(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

    def _set(self, tokenizer: RobertaTokenizer):
        self.tokenizer = tokenizer
        # block the decoder to save memory
        self.decoder = nn.Linear(2, 1)

    def forward(self, input_ids: Optional[torch.LongTensor] = None):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_outputs.last_hidden_state[:, 0, :]
        return nn.functional.normalize(hidden_state, p=2, dim=1)
