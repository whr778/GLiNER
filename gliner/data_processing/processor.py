import random
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Dict, Union
from concurrent.futures import ProcessPoolExecutor

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from .utils import pad_2d_tensor

# Abstract base class for handling data processing
class BaseProcessor(ABC):
    def __init__(self, config, tokenizer, words_splitter, labels_tokenizer = None, decoder_tokenizer = None, preprocess_text=False):
        self.config = config
        self.transformer_tokenizer = tokenizer
        self.labels_tokenizer = labels_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        self.words_splitter = words_splitter
        self.ent_token = config.ent_token
        self.sep_token = config.sep_token

        self.preprocess_text = preprocess_text

        # Check if the tokenizer has unk_token and pad_token
        self._check_and_set_special_tokens(self.transformer_tokenizer)
        if self.labels_tokenizer:
            self._check_and_set_special_tokens(self.labels_tokenizer)

    def _check_and_set_special_tokens(self, tokenizer):
        # Check for unk_token
        if tokenizer.unk_token is None:
            default_unk_token = '[UNK]'
            warnings.warn(
                f"The tokenizer is missing an 'unk_token'. Setting default '{default_unk_token}'.",
                UserWarning
            )
            tokenizer.unk_token = default_unk_token

        # Check for pad_token
        if tokenizer.pad_token is None:
            default_pad_token = '[PAD]'
            warnings.warn(
                f"The tokenizer is missing a 'pad_token'. Setting default '{default_pad_token}'.",
                UserWarning
            )
            tokenizer.pad_token = default_pad_token

    @staticmethod
    def get_dict(spans: List[Tuple[int, int, str]], classes_to_id: Dict[str, int]) -> Dict[Tuple[int, int], int]:
        dict_tag = defaultdict(int)
        for span in spans:
            if span[2] in classes_to_id:
                dict_tag[(span[0], span[1])] = classes_to_id[span[2]]
        return dict_tag

    @abstractmethod
    def preprocess_example(self, tokens: List[str], ner: List[Tuple[int, int, str]],
                         classes_to_id: Dict[str, int]) -> Dict:
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def create_labels(self) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method")
    
    @abstractmethod
    def tokenize_and_prepare_labels(self):
        pass

    @staticmethod
    def get_negatives(batch_list: List[Dict], sampled_neg: int = 5) -> List[str]:
        ent_types = []
        for b in batch_list:
            types = set([el[-1] for el in b['ner']])
            ent_types.extend(list(types))
        ent_types = list(set(ent_types))
        random.shuffle(ent_types)
        return ent_types[:sampled_neg]

    def prepare_text(self, text):
        new_text = []
        for token in text:
            if not token.strip():
                new_text.append(self.transformer_tokenizer.pad_token)
            else:
                redecoded = self.transformer_tokenizer.decode(
                                    self.transformer_tokenizer.encode(token), 
                                                    skip_special_tokens=True)
                if token!=redecoded:
                    new_text.append(self.transformer_tokenizer.unk_token)
                else:
                    new_text.append(token)
        return new_text
    
    def prepare_texts(self, texts):
        texts = [self.prepare_text(text) for text in texts]
        return texts

    def prepare_inputs(self, texts, entities, blank = None):
        input_texts = []
        prompt_lengths = []
        for id, text in enumerate(texts):
            input_text = []

            if blank is not None:
                entities_ = [blank]
            else:
                if type(entities)==dict:
                    entities_=entities
                else:
                    entities_=entities[id]
                
                if self.config.decoder_mode == 'prompt':
                    entities_ = [f"label{i}" for i in range(len(entities_))]

            for ent in entities_:
                input_text.append(self.ent_token)
                input_text.append(ent)
            input_text.append(self.sep_token)
            prompt_length = len(input_text)
            prompt_lengths.append(prompt_length)
            input_text.extend(text)
            input_texts.append(input_text)
        return input_texts, prompt_lengths
    
    def prepare_word_mask(self, texts, tokenized_inputs, prompt_lengths = None, token_level=False):
        words_masks = []
        for id in range(len(texts)):
            if prompt_lengths is not None:
                prompt_length = prompt_lengths[id]
            else:
                prompt_length = 0
            words_mask = []
            prev_word_id=None
            words_count=0
            for word_id in tokenized_inputs.word_ids(id):
                if word_id is None:
                    words_mask.append(0)
                elif word_id != prev_word_id or token_level:
                    if words_count<prompt_length:
                        words_mask.append(0)
                    else:
                        masking_word_id = word_id-prompt_length+1
                        words_mask.append(masking_word_id)
                    if word_id != prev_word_id:
                        words_count+=1
                else:
                    words_mask.append(0)
                prev_word_id = word_id
            words_masks.append(words_mask)
        return words_masks
    
    def tokenize_inputs(self, texts, entities, prepare_labels: bool = False, blank = None):

        input_texts, prompt_lengths = self.prepare_inputs(texts, entities, blank=blank)

        if self.preprocess_text:
            input_texts = self.prepare_texts(input_texts)

        tokenized_inputs = self.transformer_tokenizer(
            input_texts,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding="longest",
        )
        words_masks = self.prepare_word_mask(texts, tokenized_inputs, prompt_lengths)
        tokenized_inputs["words_mask"] = torch.tensor(words_masks)

        if self.decoder_tokenizer is not None and self.config.decoder_mode == 'span':
            decoder_input_texts = [[f" {t}" if i else t for i, t in enumerate(tokens)] for tokens in input_texts]
            decoder_tokenized_inputs = self.decoder_tokenizer(
                decoder_input_texts,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                padding="longest",
            )
            tokenized_inputs['decoder_input_ids'] = decoder_tokenized_inputs['input_ids']
            tokenized_inputs['decoder_attention_mask'] = decoder_tokenized_inputs['attention_mask']

            if self.config.full_decoder_context:
                decoder_words_masks = self.prepare_word_mask(texts, decoder_tokenized_inputs, prompt_lengths, token_level=True)
                tokenized_inputs['decoder_words_mask'] = torch.tensor(decoder_words_masks)

        if prepare_labels and self.config.decoder_mode == 'prompt':
            if isinstance(entities, dict):
                entities_ = list(entities)
            else:
                entities_ = [ent for sample_entities in entities for ent in sample_entities]

            tokenized_targets = self.decoder_tokenizer(
                entities_,
                return_tensors="pt",
                truncation=True,
                padding="longest",
            )

            target_ids = tokenized_targets["input_ids"]
            target_mask = tokenized_targets["attention_mask"]
            tokenized_inputs["decoder_labels_mask"] = target_mask

            tokenized_inputs["decoder_labels_ids"] = target_ids

            labels = target_ids.clone()
            tokenized_inputs["decoder_labels"] = labels

        return tokenized_inputs

    def batch_generate_class_mappings(self, batch_list: List[Dict], negatives: List[str]=None) -> Tuple[
        List[Dict[str, int]], List[Dict[int, str]]]:
        if negatives is None:
            negatives = self.get_negatives(batch_list, 100)
        class_to_ids = []
        id_to_classes = []
        for b in batch_list:
            max_neg_type_ratio = int(self.config.max_neg_type_ratio)
            neg_type_ratio = random.randint(0, max_neg_type_ratio) if max_neg_type_ratio else 0
            
            if "negatives" in b: # manually setting negative types
                negs_i = b["negatives"]
            else: # in-batch negative types
                negs_i = negatives[:len(b["ner"]) * neg_type_ratio] if neg_type_ratio else []

            types = list(set([el[-1] for el in b["ner"]] + negs_i))
            random.shuffle(types)
            types = types[:int(self.config.max_types)]

            if "label" in b: # labels are predefined
                types = b["label"]

            class_to_id = {k: v for v, k in enumerate(types, start=1)}
            id_to_class = {k: v for v, k in class_to_id.items()}
            class_to_ids.append(class_to_id)
            id_to_classes.append(id_to_class)

        return class_to_ids, id_to_classes

    def collate_raw_batch(self, batch_list: List[Dict], entity_types: List[Union[str, List[str]]] = None, 
                        negatives: List[str] = None, class_to_ids: Dict = None, id_to_classes: Dict = None) -> Dict:
        if entity_types is None and class_to_ids is None:
            # Generate mappings dynamically based on batch content
            class_to_ids, id_to_classes = self.batch_generate_class_mappings(batch_list, negatives)
            batch = [
                self.preprocess_example(b["tokenized_text"], b["ner"], class_to_ids[i]) 
                for i, b in enumerate(batch_list)
            ]
        else:
            if class_to_ids is None:
                # Handle cases for entity_types being a list of strings or list of lists
                if isinstance(entity_types[0], list):  # List of lists of strings
                    class_to_ids = []
                    id_to_classes = []
                    for i, types in enumerate(entity_types):
                        types = list(dict.fromkeys(types))
                        mapping = {k: v for v, k in enumerate(types, start=1)}
                        class_to_ids.append(mapping)
                        id_to_classes.append({v: k for k, v in mapping.items()})
                    batch = [
                        self.preprocess_example(b["tokenized_text"], b["ner"], class_to_ids[i]) 
                        for i, b in enumerate(batch_list)
                    ]
                else:  # Single list of strings
                    class_to_ids = {k: v for v, k in enumerate(entity_types, start=1)}
                    id_to_classes = {v: k for k, v in class_to_ids.items()}
                    batch = [
                        self.preprocess_example(b["tokenized_text"], b["ner"], class_to_ids) 
                        for b in batch_list
                    ]
            else:
                # Use provided mappings
                batch = [
                    self.preprocess_example(b["tokenized_text"], b["ner"], class_to_ids) 
                    for b in batch_list
                ]
        
        return self.create_batch_dict(batch, class_to_ids, id_to_classes)


    def collate_fn(self, batch, prepare_labels=True, *args, **kwargs):
        model_input_batch = self.tokenize_and_prepare_labels(batch, prepare_labels, *args, **kwargs)
        return model_input_batch
    
    @abstractmethod
    def create_batch_dict(self, batch: List[Dict], class_to_ids: List[Dict[str, int]],
                          id_to_classes: List[Dict[int, str]]) -> Dict:
        raise NotImplementedError("Subclasses should implement this method")

    def create_dataloader(self, data, entity_types=None, *args, **kwargs) -> DataLoader:
        return DataLoader(data, collate_fn=lambda x: self.collate_fn(x, entity_types), *args, **kwargs)


class BaseBiEncoderProcessor(BaseProcessor):
    def tokenize_inputs(self, texts, entities=None):
        if self.preprocess_text:
            texts = self.prepare_texts(texts)
            
        tokenized_inputs = self.transformer_tokenizer(texts, is_split_into_words = True, return_tensors='pt',
                                                                                truncation=True, padding="longest")

        if entities is not None:
            tokenized_labels = self.labels_tokenizer(entities, return_tensors='pt', truncation=True, padding="longest")

            tokenized_inputs['labels_input_ids'] = tokenized_labels['input_ids']
            tokenized_inputs['labels_attention_mask'] = tokenized_labels['attention_mask']

        words_masks = self.prepare_word_mask(texts, tokenized_inputs, prompt_lengths=None)
        tokenized_inputs['words_mask'] = torch.tensor(words_masks)
        return tokenized_inputs

    def batch_generate_class_mappings(self, batch_list: List[Dict], negatives: List[str]=None) -> Tuple[
        List[Dict[str, int]], List[Dict[int, str]]]:

        classes = []
        for b in batch_list:
            max_neg_type_ratio = int(self.config.max_neg_type_ratio)
            neg_type_ratio = random.randint(0, max_neg_type_ratio) if max_neg_type_ratio else 0
            
            if "negatives" in b: # manually setting negative types
                negs_i = b["negatives"]
            else: # in-batch negative types
                negs_i = []

            types = list(set([el[-1] for el in b["ner"]] + negs_i))
            

            if "label" in b: # labels are predefined
                types = b["label"]

            classes.extend(types)
        random.shuffle(classes)
        classes = list(set(classes))[:int(self.config.max_types*len(batch_list))]
        class_to_id = {k: v for v, k in enumerate(classes, start=1)}
        id_to_class = {k: v for v, k in class_to_id.items()}

        class_to_ids = [class_to_id for i in range(len(batch_list))]
        id_to_classes = [id_to_class for i in range(len(batch_list))]

        return class_to_ids, id_to_classes
    
class SpanProcessor(BaseProcessor):    
    def preprocess_example(self, tokens, ner, classes_to_id):
        if len(tokens) == 0:
            tokens = ["[PAD]"]
        max_len = self.config.max_len
        if len(tokens) > max_len:
            warnings.warn(f"Sentence of length {len(tokens)} has been truncated to {max_len}")
            tokens = tokens[:max_len]

        spans_idx = [(i, i + j) for i in range(len(tokens)) for j in range(self.config.max_width)]
        dict_lab = self.get_dict(ner, classes_to_id) if ner else defaultdict(int)
        span_label = torch.LongTensor([dict_lab[i] for i in spans_idx])
        spans_idx = torch.LongTensor(spans_idx)
        valid_span_mask = spans_idx[:, 1] > len(tokens) - 1
        span_label = span_label.masked_fill(valid_span_mask, -1)

        return {
            "tokens": tokens,
            "span_idx": spans_idx,
            "span_label": span_label,
            "seq_length": len(tokens),
            "entities": ner,
        }

    def create_batch_dict(self, batch, class_to_ids, id_to_classes):
        tokens = [el["tokens"] for el in batch]
        entities = [el["entities"] for el in batch]
        span_idx = pad_sequence([b["span_idx"] for b in batch], batch_first=True, padding_value=0)
        span_label = pad_sequence([el["span_label"] for el in batch], batch_first=True, padding_value=-1)
        seq_length = torch.LongTensor([el["seq_length"] for el in batch]).unsqueeze(-1)
        span_mask = span_label != -1

        return {
            "seq_length": seq_length,
            "span_idx": span_idx,
            "tokens": tokens,
            "span_mask": span_mask,
            "span_label": span_label,
            "entities": entities,
            "classes_to_id": class_to_ids,
            "id_to_classes": id_to_classes,
        }
    
    def create_labels(self, batch, blank = None):
        labels_batch = []
        decoder_label_strings = []
        for i in range(len(batch['tokens'])):
            tokens = batch['tokens'][i]
            classes_to_id = batch['classes_to_id']
            ner = batch['entities'][i]
            num_classes = len(classes_to_id)
            spans_idx = torch.LongTensor([
                (start, start + width)
                for start in range(len(tokens))
                for width in range(self.config.max_width)
            ])
            span_to_index = {
                (spans_idx[idx, 0].item(), spans_idx[idx, 1].item()): idx
                for idx in range(len(spans_idx))
            }
            if blank is not None:
                num_classes = 1
            labels_one_hot = torch.zeros(len(spans_idx), num_classes + 1, dtype=torch.float)
            end_token_idx = (len(tokens) - 1)
            used_spans = set()
            span_labels_dict = {}
            for (start, end, label) in ner:
                span = (start, end)
                if label in classes_to_id and span in span_to_index:
                    idx = span_to_index[span]
                    if self.config.decoder_mode == 'span':
                        class_id = classes_to_id[label] if blank is None else 1
                    else:
                        class_id = classes_to_id[label]
                    if labels_one_hot[idx, class_id] == 0 and idx not in used_spans:
                        used_spans.add(idx)
                        if end <= end_token_idx:
                            labels_one_hot[idx, class_id] = 1.0
                            span_labels_dict[idx] = label
            valid_span_mask = spans_idx[:, 1] > end_token_idx
            labels_one_hot[valid_span_mask, :] = 0.0
            labels_one_hot = labels_one_hot[:, 1:]
            labels_batch.append(labels_one_hot)
            sorted_idxs = sorted(span_labels_dict.keys())
            for idx in sorted_idxs:
                decoder_label_strings.append(span_labels_dict[idx])
        if len(labels_batch) > 1:
            labels_batch = pad_2d_tensor(labels_batch)
        else:
            labels_batch = labels_batch[0]
        decoder_tokenized_input = None
        if self.config.decoder_mode == 'span' and self.decoder_tokenizer is not None:
            if not len(decoder_label_strings):
                decoder_label_strings = ['other']
            decoder_tokenized_input = self.decoder_tokenizer(
                decoder_label_strings,
                return_tensors="pt",
                truncation=True,
                padding="longest",
                add_special_tokens=True
            )
            decoder_input_ids = decoder_tokenized_input['input_ids']
            decoder_attention_mask = decoder_tokenized_input['attention_mask']
            decoder_labels = decoder_input_ids.clone()
            decoder_labels.masked_fill(~decoder_attention_mask.bool(), -100)
            decoder_tokenized_input["labels"] = decoder_labels
        return labels_batch, decoder_tokenized_input
    
    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        if random.randint(0, 10)==10 and self.decoder_tokenizer is not None:
            blank = "entity"
        else:
            blank = None
        tokenized_input = self.tokenize_inputs(batch['tokens'], batch['classes_to_id'], prepare_labels, blank)
        if prepare_labels:
            labels, decoder_tokenized_input = self.create_labels(batch, blank=blank)
            tokenized_input['labels'] = labels

            if decoder_tokenized_input is not None:
                tokenized_input['decoder_labels_ids'] = decoder_tokenized_input['input_ids']
                tokenized_input['decoder_labels_mask'] = decoder_tokenized_input['attention_mask']
                tokenized_input['decoder_labels'] = decoder_tokenized_input['labels']

        return tokenized_input

class SpanBiEncoderProcessor(SpanProcessor, BaseBiEncoderProcessor):   
    def tokenize_and_prepare_labels(self, batch, prepare_labels, prepare_entities=True, *args, **kwargs):
        if prepare_entities:
            if type(batch['classes_to_id']) == dict:
                entities = list(batch['classes_to_id'])
            else:
                entities = list(batch['classes_to_id'][0])
        else:
            entities = None
        tokenized_input = self.tokenize_inputs(batch['tokens'], entities)
        if prepare_labels:
            labels = self.create_labels(batch)
            tokenized_input['labels'] = labels
        return tokenized_input


class TokenProcessor(BaseProcessor):
    def preprocess_example(self, tokens, ner, classes_to_id):
        # Ensure there is always a token list, even if it's empty
        if len(tokens) == 0:
            tokens = ["[PAD]"]

        # Limit the length of tokens based on configuration maximum length
        max_len = self.config.max_len
        if len(tokens) > max_len:
            warnings.warn(f"Sentence of length {len(tokens)} has been truncated to {max_len}")
            tokens = tokens[:max_len]

        # Generate entity IDs based on the NER spans provided and their classes
        try: # 'NoneType' object is not iterable
            entities_id = [[i, j, classes_to_id[k]] for i, j, k in ner if k in classes_to_id]
        except TypeError:
            entities_id = []


        example = {
            'tokens': tokens,
            'seq_length': len(tokens),
            'entities': ner,
            'entities_id': entities_id
        }
        return example

    def create_batch_dict(self, batch, class_to_ids, id_to_classes):
        # Extract relevant data from batch for batch processing
        tokens = [el["tokens"] for el in batch]
        seq_length = torch.LongTensor([el["seq_length"] for el in batch]).unsqueeze(-1)
        entities = [el["entities"] for el in batch]
        entities_id = [el["entities_id"] for el in batch]

        # Assemble and return the batch dictionary
        batch_dict = {
            "tokens": tokens,
            "seq_length": seq_length,
            "entities": entities,
            "entities_id": entities_id,
            "classes_to_id": class_to_ids,
            "id_to_classes": id_to_classes,
        }

        return batch_dict

    def create_labels(self, entities_id, batch_size, seq_len, num_classes):
        word_labels = torch.zeros(
            batch_size, seq_len, num_classes, 3, dtype=torch.float
        )

        for i, sentence_entities in enumerate(entities_id):      
            for st, ed, sp_label in sentence_entities:           
                sp_label -= 1                                   

                # skip entities that point beyond sequence length
                if st >= seq_len or ed >= seq_len:
                    continue

                word_labels[i, st, sp_label, 0] = 1              # start token
                word_labels[i, ed, sp_label, 1] = 1              # end token
                word_labels[i, st:ed + 1, sp_label, 2] = 1       # inside tokens (inclusive)

        return word_labels

    def tokenize_and_prepare_labels(self, batch, prepare_labels, *args, **kwargs):
        batch_size = len(batch['tokens'])
        seq_len = batch['seq_length'].max()
        num_classes = max([len(cid) for cid in batch['classes_to_id']])

        tokenized_input = self.tokenize_inputs(batch['tokens'], batch['classes_to_id'])
        
        if prepare_labels:
            labels = self.create_labels(batch['entities_id'], batch_size, seq_len, num_classes)
            tokenized_input['labels'] = labels
        return tokenized_input

class TokenBiEncoderProcessor(TokenProcessor, BaseBiEncoderProcessor):   
    def tokenize_and_prepare_labels(self, batch, prepare_labels, prepare_entities=True, **kwargs):
        if prepare_entities:
            if type(batch['classes_to_id']) == dict:
                entities = list(batch['classes_to_id'])
            else:
                entities = list(batch['classes_to_id'][0])
        else:
            entities = None
        batch_size = len(batch['tokens'])
        seq_len = batch['seq_length'].max()
        num_classes = len(entities)

        tokenized_input = self.tokenize_inputs(batch['tokens'], entities)
        
        if prepare_labels:
            labels = self.create_labels(batch['entities_id'], batch_size, seq_len, num_classes)
            tokenized_input['labels'] = labels

        return tokenized_input