from .ner.crf_for_ner import CrfForNer
from .ner.span_for_ner import SpanForNer
from .ner.mrc_for_ner import MRCForNer
from .ner.global_pointer_for_ner import GlobalPointerForNer


MODEL_CLS_MAP = {
    'crf_for_ner': CrfForNer,
    'span_for_ner': SpanForNer,
    'mrc_for_ner': MRCForNer,
    'global_pointer_for_ner': GlobalPointerForNer
}
