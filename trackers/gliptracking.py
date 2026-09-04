# Copyright (c) OpenMMLab. All rights reserved.
import math
import re
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor

from .center_region_iou import is_center_by_iou
from .motion_stats import compute_motion_vec_from_boxes

ForwardResults = Union[
    Dict[str, Tensor], List[DetDataSample], Tuple[Tensor, ...], Tensor
]


def find_noun_phrases(caption: str) -> list:
    """Find noun phrases in a caption using nltk.

    Args:
        caption (str): The caption to analyze.

    Returns:
        list: List of noun phrases found in the caption.

    Examples:
        >>> caption = 'There is two cat and a remote in the picture'
        >>> find_noun_phrases(caption)
        ['cat', 'a remote', 'the picture']
    """
    try:
        import nltk
    except ImportError:
        raise RuntimeError("NLTK is required; install it with `pip install nltk`.")

    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = []
    for subtree in result.subtrees():
        if subtree.label() == "NP":
            noun_phrases.append(" ".join(t[0] for t in subtree.leaves()))

    return noun_phrases


def remove_punctuation(text: str) -> str:
    """Remove punctuation from a text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with punctuation removed.
    """
    punctuation = [
        "|",
        ":",
        ";",
        "@",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "^",
        "'",
        '"',
        "’",
        "`",
        "?",
        "$",
        "%",
        "#",
        "!",
        "&",
        "*",
        "+",
        ",",
        ".",
    ]
    for p in punctuation:
        text = text.replace(p, "")
    return text.strip()


def run_ner(caption: str) -> Tuple[list, list]:
    """Run NER on a caption and return the tokens and noun phrases.

    Args:
        caption (str): The input caption.

    Returns:
        Tuple[List, List]: A tuple containing the tokens and noun phrases.
            - tokens_positive (List): A list of token positions.
            - noun_phrases (List): A list of noun phrases.
    """
    noun_phrases = find_noun_phrases(caption)
    noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
    noun_phrases = [phrase for phrase in noun_phrases if phrase != ""]

    tokens_positive = []
    for entity in noun_phrases:
        try:
            # Treat each occurrence as a distinct entity.
            for m in re.finditer(entity, caption.lower()):
                tokens_positive.append([[m.start(), m.end()]])
        except Exception:
            print("noun entities:", noun_phrases)
            print("entity:", entity)
            print("caption:", caption.lower())
    return tokens_positive, noun_phrases


def create_positive_map(
    tokenized, tokens_positive: list, max_num_entities: int = 256
) -> Tensor:
    """Construct a box-to-token association map.

    Args:
        tokenized: The tokenized input.
        tokens_positive (list): A list of token ranges
            associated with positive boxes.
        max_num_entities (int, optional): The maximum number of entities.
            Defaults to 256.

    Returns:
        torch.Tensor: The positive map.

    Raises:
        Exception: If an error occurs during token-to-char mapping.
    """
    positive_map = torch.zeros(
        (len(tokens_positive), max_num_entities), dtype=torch.float
    )

    for j, tok_list in enumerate(tokens_positive):
        for beg, end in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print("beg:", beg, "end:", end)
                print("token_positive:", tokens_positive)
                raise e
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except Exception:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except Exception:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def create_positive_map_label_to_token(positive_map: Tensor, plus: int = 0) -> dict:
    """Map label indices to their associated token indices.

    Args:
        positive_map (Tensor): The positive map tensor.
        plus (int, optional): Value added to the label for indexing.
            Defaults to 0.

    Returns:
        dict: The dictionary mapping the label to the token.
    """
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(
            positive_map[i], as_tuple=True
        )[0].tolist()
    return positive_map_label_to_token


def clean_label_name(name: str) -> str:
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name


def chunks(lst: list, n: int) -> list:
    """Split a list into consecutive chunks of at most ``n`` items."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i : i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert counter == len(lst)

    return all_


__all__ = ["GLIPTracking"]

POSITION_PHRASES = {
    "center": " near the center",
    "top-left": " near the top-left",
    "top-right": " near the top-right",
    "bottom-left": " near the bottom-left",
    "bottom-right": " near the bottom-right",
}

DIRECTION_PHRASES = {
    "right": "moving rightward",
    "down": "moving downward",
    "left": "moving leftward",
    "up": "moving upward",
}


@MODELS.register_module()
class GLIPTracking(SingleStageDetector):
    """GLIP tracker with template-search fusion and motion-aware prompts.

    This detector extends `GLIP <https://arxiv.org/abs/2112.03857>`_ with a
    dedicated feature-fusion module and adaptive prompts for UAV tracking.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        language_model (:obj:`ConfigDict` or dict): The language model config.
        motion_history_max_len (int): Maximum number of motion observations
            retained for temporal reasoning.
        fusion_module (:obj:`ConfigDict` or dict, optional): The fusion module config.
        position_hysteresis_factor (float): Hysteresis factor for center-region
            classification. Defaults to 1.2.
        direction_hysteresis_deg (float): Directional hysteresis margin in
            degrees. Defaults to 15.0.
        iou_thr (float): IoU threshold for center-region classification.
            Defaults to 0.5.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of GLIP. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of GLIP. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    POSITION_CENTER_HYSTERESIS = 1.2
    DIRECTION_HYSTERESIS_DEG = 15

    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        bbox_head: ConfigType,
        language_model: ConfigType,
        motion_history_max_len: int,
        fusion_module: OptConfigType = None,
        position_hysteresis_factor: float = POSITION_CENTER_HYSTERESIS,
        direction_hysteresis_deg: float = DIRECTION_HYSTERESIS_DEG,
        iou_thr: float = 0.5,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        self.language_model = MODELS.build(language_model)
        self._special_tokens = ". "
        self.fusion_module = MODELS.build(fusion_module)

        # Stateful priors used by sequential inference.
        self._last_diag_override = None
        self._last_pos_phrase = ""
        self._last_direction_phrase = ""
        self._last_prompt_structures: List[List[Dict[str, Optional[str]]]] = []
        self.position_hysteresis_factor = float(position_hysteresis_factor)
        self.direction_hysteresis_deg = float(direction_hysteresis_deg)
        self.iou_thr = float(iou_thr)

        # Fallback history when motion metadata is unavailable at inference.
        self._bbox_history: Dict[int, List[Tensor]] = {}
        self.motion_history_max_len = int(motion_history_max_len)

    def forward(
        self,
        T_inputs: torch.Tensor,
        S_inputs: torch.Tensor,
        data_samples: OptSampleList = None,
        mode: str = "tensor",
    ) -> ForwardResults:
        template_samples = self.transform_data_samples(data_samples, "T")
        search_samples = self.transform_data_samples(data_samples, "S")

        target_size = (
            S_inputs.shape[-2:] if S_inputs is not None else T_inputs.shape[-2:]
        )

        T_feats = self._crop_template_with_padding(
            T_inputs, template_samples, target_size
        )
        S_feats = S_inputs

        if mode == "loss":
            return self.loss(T_feats, S_feats, template_samples, search_samples)
        elif mode == "predict":
            return self.predict(T_feats, S_feats, template_samples, search_samples)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". ' "Only supports loss, predict and tensor mode"
            )

    def _process_adaptive_text_prompts(
        self, batch_data_samples: SampleList
    ) -> List[List[str]]:
        """Generate size- and motion-aware prompts for each sample."""
        prompts_per_sample: List[List[str]] = []
        for data_sample in batch_data_samples:
            prompts = self._generate_default_text_prompt(data_sample)
            prompts_per_sample.append(prompts)
        return prompts_per_sample

    def _generate_default_text_prompt(self, data_sample: DetDataSample) -> List[str]:
        """Generate adaptive text prompts from bounding-box geometry."""
        prompts: List[str] = []
        diag_override = None
        if hasattr(data_sample, "metainfo"):
            # Prefer a smoothed upstream size estimate over raw motion metadata.
            diag_override = data_sample.metainfo.get("motion_size_diag", None)
            mv = data_sample.metainfo.get("motion_vec", None)
            if mv is None:
                mv = data_sample.metainfo.get("motion", None)
            if mv is None:
                mv = data_sample.metainfo.get("motion_feat", None)
            try:
                mv_tensor = (
                    self._coerce_motion_sequence(mv, device=torch.device("cpu"))
                    if mv is not None
                    else None
                )
                if diag_override is None and mv_tensor is not None:
                    latest_mv = mv_tensor[-1]
                    bw = float(latest_mv[2])
                    bh = float(latest_mv[3])
                    # Ignore zero motion vectors as invalid size observations.
                    if abs(bw) + abs(bh) > 1e-6:
                        img_shape = data_sample.metainfo.get(
                            "S_img_shape", None
                        ) or data_sample.metainfo.get("img_shape", None)
                        if img_shape and len(img_shape) >= 2:
                            h, w = img_shape[:2]
                            diag_override = math.sqrt(
                                (max(0.0, bw) * float(w)) ** 2
                                + (max(0.0, bh) * float(h)) ** 2
                            )
                        else:
                            diag_override = math.sqrt(
                                max(0.0, bw) ** 2 + max(0.0, bh) ** 2
                            )
            except Exception:
                diag_override = None
            if diag_override is not None:
                self._last_diag_override = diag_override

        # Reuse the most recent valid size when current geometry is unavailable.
        if (
            diag_override is None
            and getattr(self, "_last_diag_override", None) is not None
        ):
            diag_override = self._last_diag_override

        if (
            (not prompts)
            and hasattr(data_sample, "gt_instances")
            and data_sample.gt_instances.bboxes is not None
            and data_sample.gt_instances.bboxes.numel() > 0
        ):
            for bbox in data_sample.gt_instances.bboxes:
                if diag_override is None:
                    x1, y1, x2, y2 = bbox.tolist()
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    diag_override = math.sqrt(w * w + h * h)
                    self._last_diag_override = diag_override
                prompts.append(
                    self._generate_adaptive_prompt_from_bbox(
                        bbox, diag_override=diag_override
                    )
                )
        prompts = [p for p in prompts if p]
        if not prompts:
            prompts = ["uav ."]
        return prompts

    def _generate_adaptive_prompt_from_bbox(
        self, bbox: Tensor, diag_override=None
    ) -> str:
        """Describe target scale from a box or a smoothed diagonal."""
        if diag_override is not None:
            diag = float(diag_override)
        else:
            x1, y1, x2, y2 = bbox.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            diag = math.sqrt(w * w + h * h)

        if diag < 10:
            return "a tiny uav ."
        elif diag < 30:
            return "a small uav ."
        elif diag < 50:
            return "a typical uav ."
        else:
            return "a large uav ."

    @staticmethod
    def _prompts_all_equal(prompts: List[List[str]]) -> bool:
        """Return whether every sample uses identical prompts."""
        if not prompts:
            return True
        reference = tuple(prompts[0])
        return all(tuple(p) == reference for p in prompts[1:])

    def transform_data_samples(
        self, data_samples: OptSampleList = None, mode: str = None
    ):
        """Deep-copy samples and expose branch-prefixed fields as aliases."""
        new_samples = []
        if mode == "T":
            for sample in data_samples:
                new_sample = deepcopy(sample)
                new_metainfo = {}

                for key, value in new_sample.metainfo.items():
                    if key.startswith("T_"):
                        new_key = key[2:]
                        new_metainfo[new_key] = value

                if hasattr(new_sample, "gt_instances"):
                    new_gt_instances = deepcopy(new_sample.gt_instances)

                    for key in new_sample.gt_instances.keys():
                        if key.startswith("T_"):
                            new_key = key[2:]
                            new_gt_instances[new_key] = new_sample.gt_instances[key]

                    new_sample.gt_instances = new_gt_instances

                if hasattr(new_sample, "ignored_instances"):
                    new_ignored_instances = deepcopy(new_sample.ignored_instances)

                    for key in new_sample.ignored_instances.keys():
                        if key.startswith("T_"):
                            new_key = key[2:]
                            new_ignored_instances[
                                new_key
                            ] = new_sample.ignored_instances[key]

                    new_sample.ignored_instances = new_ignored_instances

                for key, value in new_metainfo.items():
                    new_sample.set_metainfo({key: value})

                new_samples.append(new_sample)
        else:
            for sample in data_samples:
                new_sample = deepcopy(sample)
                new_metainfo = {}

                for key, value in new_sample.metainfo.items():
                    if key.startswith("S_"):
                        new_key = key[2:]
                        new_metainfo[new_key] = value

                if hasattr(new_sample, "gt_instances"):
                    new_gt_instances = deepcopy(new_sample.gt_instances)

                    for key in new_sample.gt_instances.keys():
                        if key.startswith("S_"):
                            new_key = key[2:]
                            new_gt_instances[new_key] = new_sample.gt_instances[key]

                    new_sample.gt_instances = new_gt_instances

                if hasattr(new_sample, "ignored_instances"):
                    new_ignored_instances = deepcopy(new_sample.ignored_instances)

                    for key in new_sample.ignored_instances.keys():
                        if key.startswith("S_"):
                            new_key = key[2:]
                            new_ignored_instances[
                                new_key
                            ] = new_sample.ignored_instances[key]

                    new_sample.ignored_instances = new_ignored_instances

                for key, value in new_metainfo.items():
                    new_sample.set_metainfo({key: value})

                new_samples.append(new_sample)
        return new_samples

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ""
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if "prefix" in enhanced_text_dict:
                    caption_string += enhanced_text_dict["prefix"]
                start_i = len(caption_string)
                if "name" in enhanced_text_dict:
                    caption_string += enhanced_text_dict["name"]
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if "suffix" in enhanced_text_dict:
                    caption_string += enhanced_text_dict["suffix"]
            else:
                tokens_positive.append(
                    [[len(caption_string), len(caption_string) + len(word)]]
                )
                caption_string += word

            if idx != len(original_caption) - 1:
                caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ""
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string), len(caption_string) + len(word)]]
            )
            caption_string += word
            if idx != len(original_caption) - 1:
                caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None,
    ) -> Tuple[dict, str, list, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts
                )
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption
                )

            tokenized = self.language_model.tokenizer(
                [caption_string], return_tensors="pt"
            )
            entities = original_caption
        else:
            original_caption = original_caption.strip(self._special_tokens)
            tokenized = self.language_model.tokenizer(
                [original_caption], return_tensors="pt"
            )
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(tokenized, tokens_positive)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1
        )
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith("."):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith("."):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption], return_tensors="pt"
                )
                positive_map_label_to_token, positive_map = self.get_positive_map(
                    tokenized, tokens_positive
                )

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0] : t[1]])
                    entities.append(" / ".join(instance_entities))
                return (
                    positive_map_label_to_token,
                    original_caption,
                    positive_map,
                    entities,
                )

        chunked_size = self.test_cfg.get("chunked_size", -1)
        if not self.training and chunked_size > 0:
            assert (
                isinstance(original_caption, (list, tuple)) or custom_entities is True
            )
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt
            )
            (
                positive_map_label_to_token,
                caption_string,
                positive_map,
                entities,
            ) = all_output
        else:
            (
                tokenized,
                caption_string,
                tokens_positive,
                entities,
            ) = self.get_tokens_and_prompts(
                original_caption, custom_entities, enhanced_text_prompt
            )
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive
            )
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn(
                    "Inputting a text that is too long will result "
                    "in poor prediction performance. "
                    "Please reduce the text length."
                )
        return positive_map_label_to_token, caption_string, positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
        self,
        original_caption: Union[list, tuple],
        enhanced_text_prompts: Optional[ConfigType] = None,
    ):
        chunked_size = self.test_cfg.get("chunked_size", -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(list(range(1, len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts
                )
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i]
                )
            tokenized = self.language_model.tokenizer(
                [caption_string], return_tensors="pt"
            )
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn(
                    "Inputting a text that is too long will result "
                    "in poor prediction performance. "
                    "Please reduce the --chunked-size."
                )
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive
            )

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return (
            positive_map_label_to_token_chunked,
            caption_string_chunked,
            positive_map_chunked,
            entities_chunked,
        )

    @staticmethod
    def _coerce_motion_sequence(
        value, device: torch.device, motion_dim: int = 4
    ) -> Optional[Tensor]:
        """Convert user/pipeline motion input to a (T, motion_dim) tensor."""
        if value is None:
            return None
        if isinstance(value, Tensor):
            seq = value.to(device=device, dtype=torch.float32)
        else:
            seq = torch.as_tensor(value, device=device, dtype=torch.float32)
        if seq.numel() == 0:
            return None
        if seq.dim() == 1:
            if seq.numel() % motion_dim == 0:
                seq = seq.reshape(-1, motion_dim)
            else:
                seq = seq.flatten()
                if seq.numel() < motion_dim:
                    seq = F.pad(seq, (0, motion_dim - seq.numel()), value=0.0)
                else:
                    seq = seq[:motion_dim]
                seq = seq.unsqueeze(0)
        elif seq.dim() == 2:
            if seq.shape[-1] != motion_dim:
                if seq.numel() % motion_dim != 0:
                    return None
                seq = seq.reshape(-1, motion_dim)
        else:
            if seq.numel() % motion_dim != 0:
                return None
            seq = seq.reshape(-1, motion_dim)
        return seq

    def _extract_motion_vectors(
        self,
        data_samples: SampleList,
        device: torch.device,
        motion_dim: int = 4,
        return_sequence: bool = True,
    ) -> torch.Tensor:
        """Extract temporal motion features.

        Explicit motion metadata from the data pipeline or inferencer takes
        precedence. Prediction history is used only as an online fallback.

        Args:
            data_samples: Batch of tracking data samples.
            device: Device for the returned tensor.
            motion_dim: Motion-vector dimension, conventionally
                ``[cx, cy, width, height]``.
            return_sequence: Return a padded temporal sequence when ``True``;
                otherwise return a temporal mean for each sample.

        Returns:
            Motion tensor with shape ``(B, T, motion_dim)`` or
            ``(B, motion_dim)``.
        """
        motion_list = []
        max_seq_len = 1

        for idx, sample in enumerate(data_samples):
            img_shape = None
            meta = getattr(sample, "metainfo", None)
            if isinstance(meta, dict):
                img_shape = meta.get("S_img_shape", None) or meta.get("img_shape", None)

            explicit_motion = None
            if isinstance(meta, dict):
                for key in ("motion_vec", "motion", "motion_feat"):
                    explicit_motion = meta.get(key, None)
                    if explicit_motion is not None:
                        break
            if explicit_motion is None and hasattr(sample, "motion_vec"):
                explicit_motion = getattr(sample, "motion_vec")

            motion_seq = self._coerce_motion_sequence(
                explicit_motion, device, motion_dim
            )

            if motion_seq is None:
                sample_id = idx
                if isinstance(meta, dict):
                    sample_id = meta.get("track_id", meta.get("img_id", idx))
                bbox_history = self._bbox_history.get(sample_id, [])
                if len(bbox_history) == 0:
                    motion_seq = torch.zeros((1, motion_dim), device=device)
                else:
                    boxes = torch.stack(bbox_history, dim=0).to(device)
                    motion_seq = compute_motion_vec_from_boxes(
                        boxes,
                        motion_history_max_len=self.motion_history_max_len,
                        img_shape=img_shape,
                        return_sequence=True,
                    ).to(device)

            if motion_seq.shape[0] > self.motion_history_max_len:
                motion_seq = motion_seq[-self.motion_history_max_len :]

            max_seq_len = max(max_seq_len, motion_seq.shape[0])
            motion_list.append(motion_seq)

        if not motion_list:
            if return_sequence:
                return torch.zeros((0, 1, motion_dim), device=device)
            return torch.zeros((0, motion_dim), device=device)

        if not return_sequence:
            aggregated = [seq.mean(dim=0) for seq in motion_list]
            return torch.stack(aggregated, dim=0)

        padded_list = []
        for seq in motion_list:
            seq_len = seq.shape[0]
            if seq_len < max_seq_len:
                # Left padding keeps the most recent observations at the end.
                pad_len = max_seq_len - seq_len
                padding = torch.zeros(pad_len, motion_dim, device=device)
                seq = torch.cat([padding, seq], dim=0)
            padded_list.append(seq)

        return torch.stack(padded_list, dim=0)

    def _update_bbox_history(
        self, data_samples: SampleList, pred_bboxes: Optional[List[Tensor]] = None
    ) -> None:
        """Update the bounding-box history used for online fallback."""
        for idx, sample in enumerate(data_samples):
            meta = getattr(sample, "metainfo", None)

            sample_id = idx
            if isinstance(meta, dict):
                sample_id = meta.get("track_id", meta.get("img_id", idx))

            bbox = None
            if pred_bboxes is not None and idx < len(pred_bboxes):
                preds = pred_bboxes[idx]
                if preds is not None and preds.numel() > 0:
                    bbox = (
                        preds[0, :4].detach().cpu()
                        if preds.dim() == 2
                        else preds[:4].detach().cpu()
                    )
            elif hasattr(sample, "pred_instances"):
                pred_instances = sample.pred_instances
                if (
                    hasattr(pred_instances, "bboxes")
                    and pred_instances.bboxes is not None
                ):
                    if pred_instances.bboxes.numel() > 0:
                        bbox = pred_instances.bboxes[0].detach().cpu()

            if bbox is None:
                continue

            bbox = bbox.flatten()[:4]
            self._bbox_history.setdefault(sample_id, []).append(bbox)
            if len(self._bbox_history[sample_id]) > self.motion_history_max_len:
                self._bbox_history[sample_id] = self._bbox_history[sample_id][
                    -self.motion_history_max_len :
                ]

    def reset_tracking_state(self) -> None:
        """Reset all stateful priors before processing a new sequence."""
        self._bbox_history.clear()
        self._last_diag_override = None
        self._last_pos_phrase = ""
        self._last_direction_phrase = ""
        self._last_prompt_structures = []

    @staticmethod
    def _prompt_text_to_structure(prompt: str) -> Dict[str, Optional[str]]:
        """Convert a final text prompt into serializable semantic fields."""
        raw_prompt = str(prompt)
        normalized_prompt = " ".join(raw_prompt.split())
        lower_prompt = normalized_prompt.lower()

        scale = None
        for scale_name in ("tiny", "small", "typical", "large"):
            if re.search(rf"\b{scale_name}\s+uav\b", lower_prompt):
                scale = scale_name
                break

        class_name = "uav" if re.search(r"\buav\b", lower_prompt) else None

        position = None
        for position_name, phrase in POSITION_PHRASES.items():
            if phrase.strip() in lower_prompt:
                position = position_name
                break

        direction = None
        for direction_name, phrase in DIRECTION_PHRASES.items():
            if phrase in lower_prompt:
                direction = direction_name
                break

        return {
            "prompt": raw_prompt,
            "normalized_prompt": normalized_prompt,
            "scale": scale,
            "class": class_name,
            "position": position,
            "direction": direction,
        }

    @classmethod
    def _build_prompt_structures(cls, prompts) -> List[List[Dict[str, Optional[str]]]]:
        """Build a ``batch -> prompts -> structure`` representation."""
        batch_structures = []
        for sample_prompts in prompts:
            if isinstance(sample_prompts, (list, tuple)):
                prompt_list = sample_prompts
            else:
                prompt_list = [sample_prompts]
            batch_structures.append(
                [
                    cls._prompt_text_to_structure(prompt)
                    for prompt in prompt_list
                    if prompt is not None
                ]
            )
        return batch_structures

    def get_last_prompt_structures(self) -> List[List[Dict[str, Optional[str]]]]:
        """Return a copy of the prompt structures used for last inference."""
        return deepcopy(self._last_prompt_structures)

    def _compute_position_phrase(self, motion_vec: Tensor) -> str:
        """Map normalized target geometry to a position phrase.

        Args:
            motion_vec: Normalized ``[cx, cy, width, height]`` geometry with
                shape ``(4,)`` or ``(K, 4)``.

        Returns:
            A position phrase, or an empty string for invalid input.
        """
        if motion_vec.dim() == 2:
            mv = motion_vec[-1]
        else:
            mv = motion_vec

        if mv.numel() < 2:
            return ""

        if not torch.isfinite(mv).all():
            return ""
        if mv.norm().item() < 1e-4:
            return ""

        cx, cy = float(mv[0]), float(mv[1])
        bw = float(mv[2]) if mv.numel() >= 3 else 0.0
        bh = float(mv[3]) if mv.numel() >= 4 else 0.0

        dx = cx - 0.5
        dy = cy - 0.5

        effective_iou_thr = self.iou_thr
        if not self.training and "center" in self._last_pos_phrase:
            effective_iou_thr = self.iou_thr / self.position_hysteresis_factor

        if is_center_by_iou(cx, cy, bw, bh, iou_thr=effective_iou_thr):
            pos = POSITION_PHRASES["center"]
        else:
            v_pos = "top" if dy < 0 else "bottom"
            h_pos = "left" if dx < 0 else "right"
            pos = POSITION_PHRASES[f"{v_pos}-{h_pos}"]

        if not self.training:
            self._last_pos_phrase = pos

        return pos

    def _angle_to_direction_4way(self, angle: float) -> str:
        """Map an angle to one of four direction phrases with hysteresis.

        The four 90-degree sectors are centered at right (0), down (90),
        left (180), and up (-90). Image coordinates use a downward Y-axis.

        Args:
            angle: Motion angle in degrees.

        Returns:
            Direction phrase for the selected sector.
        """
        directions = {
            DIRECTION_PHRASES["right"]: 0.0,
            DIRECTION_PHRASES["down"]: 90.0,
            DIRECTION_PHRASES["left"]: 180.0,
            DIRECTION_PHRASES["up"]: -90.0,
        }

        half_sector = 45.0

        def angle_diff(a1: float, a2: float) -> float:
            """Return the smallest absolute difference between two angles."""
            diff = abs(a1 - a2)
            if diff > 180:
                diff = 360 - diff
            return diff

        if not self.training:
            last_dir = getattr(self, "_last_direction_phrase", "")
            if last_dir and last_dir in directions:
                center = directions[last_dir]
                diff = angle_diff(angle, center)
                if diff <= (half_sector + self.direction_hysteresis_deg):
                    return last_dir

        best_dir = DIRECTION_PHRASES["right"]
        min_diff = 360.0
        for name, center in directions.items():
            diff = angle_diff(angle, center)
            if diff < min_diff:
                min_diff = diff
                best_dir = name

        if not self.training:
            self._last_direction_phrase = best_dir

        return best_dir

    def _compute_direction_phrase(self, motion_seq: Tensor) -> str:
        """Estimate a direction phrase from normalized motion history.

        Args:
            motion_seq: History with shape ``(K, 4)`` containing normalized
                ``[cx, cy, width, height]`` observations.

        Returns:
            Direction phrase, or an empty string when motion is insufficient.
        """
        if motion_seq is None:
            return ""

        valid_mask = motion_seq[:, 2] > 1e-6
        valid_seq = motion_seq[valid_mask]

        if valid_seq.shape[0] < 2:
            return ""

        if valid_seq.shape[0] > self.motion_history_max_len:
            valid_seq = valid_seq[-self.motion_history_max_len :]

        start_vec = valid_seq[0]
        end_vec = valid_seq[-1]
        dx = float(end_vec[0] - start_vec[0])
        dy = float(end_vec[1] - start_vec[1])

        norm = math.sqrt(dx * dx + dy * dy)

        if norm < 1e-6:
            return ""

        angle = math.atan2(dy, dx) * 180.0 / math.pi
        return self._angle_to_direction_4way(angle)

    def _augment_prompts_with_motion(
        self, prompts: List[str], motion_vecs: Tensor
    ) -> List[str]:
        """Append position and direction phrases to base prompts.

        Args:
            prompts: Base text prompts.
            motion_vecs: Motion data with shape ``(B, 4)`` or ``(B, T, 4)``.

        Returns:
            Motion-aware prompts.
        """
        if motion_vecs is None or len(prompts) == 0:
            return prompts

        augmented = []
        for prompt, mv in zip(prompts, motion_vecs):
            if isinstance(prompt, (list, tuple)):
                augmented.append(
                    [
                        self._augment_prompts_with_motion([p], mv.unsqueeze(0))[0]
                        for p in prompt
                    ]
                )
                continue

            if prompt is None:
                augmented.append(prompt)
                continue

            phrase_pos = self._compute_position_phrase(mv)

            phrase_dir = ""
            if mv.dim() == 2 and mv.shape[0] >= 2:
                phrase_dir = self._compute_direction_phrase(mv)

            parts = []
            if phrase_pos:
                parts.append(phrase_pos)
            if phrase_dir:
                parts.append(phrase_dir)

            suffix = ", ".join(parts)

            if suffix:
                base = prompt.rstrip().rstrip(".")
                augmented.append(f"{base}, {suffix}.")
            else:
                augmented.append(prompt)

        return augmented

    def _crop_template_with_padding(
        self,
        template_inputs: torch.Tensor,
        template_samples: Optional[SampleList],
        target_size: Tuple[int, int],
        context_amount: float = 0.5,
        zero_clip_eps: float = 1e-6,
        min_roi_size: int = 16,
        min_context_ratio: float = 0.5,
        max_context_ratio: float = 3.0,
    ) -> torch.Tensor:
        """Crop and resize a context-aware template ROI.

        Small regions retain additional context and are padded by replication
        before nearest-neighbor resizing to avoid zero-filled boundaries.
        """

        def _bbox_to_xyxy(
            bbox: torch.Tensor, img_w: int, img_h: int
        ) -> Tuple[float, float, float, float]:
            """Convert an ``xyxy`` or ``xywh`` box to clipped ``xyxy``."""
            x1_raw, y1_raw, x2_or_w, y2_or_h = bbox.tolist()
            if x2_or_w > x1_raw and y2_or_h > y1_raw:
                x2_box, y2_box = x2_or_w, y2_or_h
            else:
                x2_box = x1_raw + max(x2_or_w, 1.0)
                y2_box = y1_raw + max(y2_or_h, 1.0)

            x1_box = max(0.0, min(x1_raw, img_w - 1.0))
            y1_box = max(0.0, min(y1_raw, img_h - 1.0))
            x2_box = max(x1_box + 1.0, min(x2_box, img_w * 1.0))
            y2_box = max(y1_box + 1.0, min(y2_box, img_h * 1.0))
            return x1_box, y1_box, x2_box, y2_box

        if template_samples is None:
            return template_inputs

        batch_size = template_inputs.shape[0]
        if len(template_samples) != batch_size:
            warnings.warn(
                "Template samples count mismatches template inputs; "
                "skip template cropping."
            )
            return template_inputs

        out_h, out_w = target_size
        processed: List[torch.Tensor] = []
        fallback_warned = False

        for idx in range(batch_size):
            template_img = template_inputs[idx]
            _, h, w = template_img.shape

            bbox_tensor = None
            sample = template_samples[idx]
            if hasattr(sample, "gt_instances"):
                gt_instances = sample.gt_instances
                if (
                    hasattr(gt_instances, "T_bboxes")
                    and gt_instances.T_bboxes is not None
                    and gt_instances.T_bboxes.numel() > 0
                ):
                    bbox_tensor = gt_instances.T_bboxes
                elif (
                    hasattr(gt_instances, "bboxes")
                    and gt_instances.bboxes is not None
                    and gt_instances.bboxes.numel() > 0
                ):
                    bbox_tensor = gt_instances.bboxes

            if bbox_tensor is None:
                cropped = template_img
            else:
                x1_box, y1_box, x2_box, y2_box = _bbox_to_xyxy(bbox_tensor[0], w, h)
                w_box = max(x2_box - x1_box, 1.0)
                h_box = max(y2_box - y1_box, 1.0)

                diag = math.sqrt(w_box * h_box)
                adaptive_ctx = 50.0 / max(diag, 1e-3)
                adaptive_ctx = max(
                    min_context_ratio, min(max_context_ratio, adaptive_ctx)
                )
                eff_context = max(context_amount, adaptive_ctx)

                p = eff_context * (w_box + h_box)
                w_context = w_box + p
                h_context = h_box + p
                side = math.sqrt(w_context * h_context)
                half_side = side / 2.0
                x_center = x1_box + w_box / 2.0
                y_center = y1_box + h_box / 2.0

                x1 = int(max(0, math.floor(x_center - half_side)))
                y1 = int(max(0, math.floor(y_center - half_side)))
                x2 = int(min(w, math.ceil(x_center + half_side)))
                y2 = int(min(h, math.ceil(y_center + half_side)))

                if x1 >= x2 or y1 >= y2:
                    cropped = template_img
                else:
                    cropped = template_img[:, y1:y2, x1:x2]

            roi_h, roi_w = cropped.shape[-2:]
            if roi_h < min_roi_size or roi_w < min_roi_size:
                pad_h = max(0, min_roi_size - roi_h)
                pad_w = max(0, min_roi_size - roi_w)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                cropped = F.pad(
                    cropped.unsqueeze(0),
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="replicate",
                ).squeeze(0)
                roi_h, roi_w = cropped.shape[-2:]
            if roi_h <= 0 or roi_w <= 0:
                processed.append(
                    F.interpolate(
                        template_img.unsqueeze(0), size=(out_h, out_w), mode="nearest"
                    ).squeeze(0)
                )
                continue

            roi_sum = cropped.abs().sum().item()
            if roi_h <= out_h and roi_w <= out_w:
                pad_top = (out_h - roi_h) // 2
                pad_bottom = out_h - roi_h - pad_top
                pad_left = (out_w - roi_w) // 2
                pad_right = out_w - roi_w - pad_left
                resized = F.pad(
                    cropped.unsqueeze(0),
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="replicate",
                ).squeeze(0)
            else:
                scale = min(out_h / roi_h, out_w / roi_w)
                new_h = max(1, int(round(roi_h * scale)))
                new_w = max(1, int(round(roi_w * scale)))
                downsampled = F.interpolate(
                    cropped.unsqueeze(0), size=(new_h, new_w), mode="nearest"
                ).squeeze(0)
                pad_top = (out_h - new_h) // 2
                pad_bottom = out_h - new_h - pad_top
                pad_left = (out_w - new_w) // 2
                pad_right = out_w - new_w - pad_left
                resized = F.pad(
                    downsampled.unsqueeze(0),
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="replicate",
                ).squeeze(0)

            resized_sum = resized.abs().sum().item()
            if getattr(self, "debug_template_crop", False):
                print(
                    f"[template-crop] idx={idx} roi_sum={roi_sum:.4f} "
                    f"resized_sum={resized_sum:.4f}",
                    flush=True,
                )

            if (
                resized_sum <= zero_clip_eps
                and template_img.abs().sum().item() > zero_clip_eps
            ):
                if not fallback_warned:
                    warnings.warn(
                        "Template crop becomes zero; fallback to resized template."
                    )
                    fallback_warned = True
                fallback = F.interpolate(
                    template_img.unsqueeze(0), size=(out_h, out_w), mode="nearest"
                ).squeeze(0)
                processed.append(fallback)
                continue

            processed.append(resized)

        return torch.stack(processed, dim=0)

    def _prepare_visual_features(
        self, T_inputs: torch.Tensor, S_inputs: torch.Tensor
    ) -> Sequence[Tensor]:
        """Extract and fuse template/search multi-scale features."""
        template_features = self.extract_feat(T_inputs)
        search_features = self.extract_feat(S_inputs)
        fused_features = self.fusion_module(template_features, search_features)
        return fused_features

    def _extract_and_fuse_features(
        self, T_inputs: torch.Tensor, S_inputs: torch.Tensor
    ) -> Tuple[Tensor, ...]:
        """Return fused multi-scale visual features as a tuple."""

        fused_features = self._prepare_visual_features(T_inputs, S_inputs)
        return tuple(fused_features)

    def loss(
        self,
        T_inputs: torch.Tensor,
        S_inputs: torch.Tensor,
        template_samples: SampleList,
        search_samples: SampleList,
    ) -> Union[dict, list]:
        """Compute losses from fused visual features and adaptive prompts."""

        visual_features = self._prepare_visual_features(T_inputs, S_inputs)

        template_text_prompts = self._process_adaptive_text_prompts(template_samples)

        gt_labels = [sample.gt_instances.labels for sample in search_samples]

        text_for_language: List[str] = []
        positive_maps: List[Tensor] = []

        motion_vecs = self._extract_motion_vectors(search_samples, T_inputs.device)

        motion_aug_prompts = self._augment_prompts_with_motion(
            template_text_prompts, motion_vecs
        )

        for prompt_entry, gt_label in zip(motion_aug_prompts, gt_labels):
            tokenized, caption_string, tokens_positive, _ = self.get_tokens_and_prompts(
                prompt_entry, True
            )
            new_tokens_positive = [tokens_positive[label] for label in gt_label]
            _, positive_map = self.get_positive_map(tokenized, new_tokens_positive)
            positive_maps.append(positive_map)
            text_for_language.append(caption_string)

        language_dict_features = self.language_model(text_for_language)

        for sample, positive_map, t_prompts in zip(
            search_samples,
            positive_maps,
            motion_aug_prompts,
        ):
            sample.set_metainfo(dict(template_text=t_prompts))
            positive_map = positive_map.to(T_inputs.device).bool().float()
            sample.gt_instances.positive_maps = positive_map

        losses = self.bbox_head.loss(
            visual_features, language_dict_features, search_samples
        )

        return losses

    def predict(
        self,
        T_inputs: torch.Tensor,
        S_inputs: torch.Tensor,
        template_samples: SampleList,
        search_samples: SampleList,
        rescale: bool = True,
    ) -> SampleList:
        """Predict tracking results from template and search inputs.

        Args:
            T_inputs (torch.Tensor): Template inputs with shape (N, C, H, W).
            S_inputs (torch.Tensor): Search inputs with shape (N, C, H, W).
            template_samples (List[:obj:`DetDataSample`]): Template-side
                samples providing template prompts and optional token spans.
            search_samples (List[:obj:`DetDataSample`]): Search-side samples;
                predictions are attached to these samples.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - label_names (List[str]): Label names of bboxes.
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        visual_features = self._extract_and_fuse_features(T_inputs, S_inputs)

        motion_vecs = self._extract_motion_vectors(search_samples, T_inputs.device)

        # Expose the latest motion state to adaptive prompt generation.
        for idx, sample in enumerate(search_samples):
            if idx < motion_vecs.shape[0]:
                mv = motion_vecs[idx]
                if mv.dim() == 2:
                    mv = mv[-1]
                sample.set_metainfo({"motion_vec": mv.detach().cpu()})

        def _maybe_init_diag_from_template():
            if getattr(self, "_last_diag_override", None) is not None:
                return
            if not template_samples:
                return
            try:
                tmpl_bbox = None
                if hasattr(template_samples[0], "gt_instances"):
                    if (
                        hasattr(template_samples[0].gt_instances, "T_bboxes")
                        and template_samples[0].gt_instances.T_bboxes is not None
                        and template_samples[0].gt_instances.T_bboxes.numel() > 0
                    ):
                        tmpl_bbox = template_samples[0].gt_instances.T_bboxes[0]
                    elif (
                        hasattr(template_samples[0].gt_instances, "bboxes")
                        and template_samples[0].gt_instances.bboxes is not None
                        and template_samples[0].gt_instances.bboxes.numel() > 0
                    ):
                        tmpl_bbox = template_samples[0].gt_instances.bboxes[0]
                if tmpl_bbox is not None:
                    x1, y1, x2, y2 = tmpl_bbox.tolist()
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    self._last_diag_override = math.sqrt(w * w + h * h)
                    for tmpl in template_samples:
                        if hasattr(tmpl, "metainfo") and tmpl.metainfo is not None:
                            tmpl.metainfo["motion_size_diag"] = self._last_diag_override
            except Exception:
                pass

        _maybe_init_diag_from_template()

        # Update the smoothed target-size prior from current motion geometry.
        for s_sample, t_sample in zip(search_samples, template_samples):
            diag_override = None
            if hasattr(s_sample, "metainfo"):
                mv = s_sample.metainfo.get("motion_vec", None)

                try:
                    if mv is not None and hasattr(mv, "__len__") and len(mv) >= 4:
                        bw = float(mv[2])
                        bh = float(mv[3])
                        if abs(bw) + abs(bh) > 1e-6:
                            img_shape = s_sample.metainfo.get(
                                "S_img_shape", None
                            ) or s_sample.metainfo.get("img_shape", None)
                            if img_shape and len(img_shape) >= 2:
                                h, w = img_shape[:2]
                                diag_override = math.sqrt(
                                    (max(0.0, bw) * float(w)) ** 2
                                    + (max(0.0, bh) * float(h)) ** 2
                                )
                            else:
                                diag_override = math.sqrt(
                                    max(0.0, bw) ** 2 + max(0.0, bh) ** 2
                                )
                except Exception:
                    diag_override = None

            if (
                diag_override is None
                and getattr(self, "_last_diag_override", None) is not None
            ):
                diag_override = self._last_diag_override

            if diag_override is not None:
                self._last_diag_override = diag_override
                if hasattr(s_sample, "metainfo"):
                    s_sample.metainfo["motion_size_diag"] = diag_override
                if hasattr(t_sample, "metainfo") and t_sample.metainfo is not None:
                    t_sample.metainfo["motion_size_diag"] = diag_override

        text_prompt_inputs = self._process_adaptive_text_prompts(template_samples)

        # Preserve the size prior when the current motion tensor is all zeros.
        try:
            if (
                motion_vecs.numel() > 0
                and float(motion_vecs.abs().sum()) < 1e-6
                and getattr(self, "_last_diag_override", None) is not None
            ):
                for tmpl in template_samples:
                    if hasattr(tmpl, "metainfo") and tmpl.metainfo is not None:
                        tmpl.metainfo["motion_size_diag"] = self._last_diag_override
        except Exception:
            pass

        text_prompt_inputs = self._augment_prompts_with_motion(
            text_prompt_inputs, motion_vecs
        )

        enhanced_text_prompts = []
        for data_sample in template_samples:
            if "caption_prompt" in data_sample:
                enhanced_text_prompts.append(data_sample.caption_prompt)
            else:
                enhanced_text_prompts.append(None)

        if "custom_entities" in template_samples[0]:
            # A batch is expected to share one custom-entity setting.
            custom_entities = template_samples[0].custom_entities
        else:
            custom_entities = False

        if self._prompts_all_equal(text_prompt_inputs):
            # Reuse tokenization when every sample has identical prompts.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompt_inputs[0],
                    custom_entities,
                    enhanced_text_prompts[0],
                    None,
                )
            ] * len(T_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompt_input, custom_entities, enhanced_text_prompt, None
                )
                for text_prompt_input, enhanced_text_prompt in zip(
                    text_prompt_inputs, enhanced_text_prompts
                )
            ]

        token_positive_maps, processed_text_prompts, _, entities = zip(
            *_positive_maps_and_prompts
        )

        # Store the exact strings subsequently passed to the language model.
        self._last_prompt_structures = self._build_prompt_structures(
            processed_text_prompts
        )

        if isinstance(processed_text_prompts[0], list):
            # Chunked prompts currently support only a batch size of one.
            assert len(T_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(processed_text_prompts[0])):
                text_prompts_once = [processed_text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                language_dict_features = self.language_model(text_prompts_once)
                search_samples[0].token_positive_map = token_positive_maps_once

                pred_instances = self.bbox_head.predict(
                    deepcopy(visual_features),
                    language_dict_features,
                    search_samples,
                    rescale=rescale,
                )[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
        else:
            language_dict_features = self.language_model(list(processed_text_prompts))

            for i, data_sample in enumerate(search_samples):
                data_sample.token_positive_map = token_positive_maps[i]

            results_list = self.bbox_head.predict(
                visual_features, language_dict_features, search_samples, rescale=rescale
            )

        for data_sample, pred_instances, entity in zip(
            search_samples, results_list, entities
        ):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if labels >= len(entity):
                        warnings.warn(
                            "The unexpected output indicates an issue with "
                            "named entity recognition. You can try "
                            "setting custom_entities=True and running "
                            "again to see if it helps."
                        )
                        label_names.append("unobject")
                    else:
                        label_names.append(entity[labels])
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances

        self._update_bbox_history(search_samples)

        return search_samples
