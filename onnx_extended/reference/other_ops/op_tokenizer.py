import re
import numpy as np
from onnx.reference.op_run import OpRun


class Tokenizer(OpRun):
    op_domain = "com.microsoft"

    def _run(
        self,
        text,
        mark=None,
        mincharnum=None,
        pad_value=None,
        separators=None,
        tokenexp=None,
        tokenexpsplit=None,
        stopwords=None,
    ):
        char_tokenization_ = tokenexp == "." or list(separators or []) == [""]
        stops_ = set(stopwords or [])
        try:
            str_separators_ = set(_ for _ in (separators or ""))
        except AttributeError as e:
            raise TypeError(f"Unable to interpret separators {separators!r}.") from e
        if tokenexp not in (None, ""):
            tokenexp_ = re.compile(tokenexp)

        if char_tokenization_:
            return self._run_char_tokenization(text, stops_, mark, pad_value)
        if str_separators_ is not None and len(str_separators_) > 0:
            str_separators = [re.compile(s) for s in str_separators_]
            return self._run_sep_tokenization(
                text, stops_, str_separators, mark, pad_value
            )
        if tokenexp not in (None, ""):
            return self._run_regex_tokenization(
                text, stops_, tokenexp_, tokenexpsplit, mark, pad_value
            )
        raise RuntimeError(
            "Unable to guess which tokenization to use, sep={}, "
            "tokenexp='{}'.".format(separators, tokenexp)
        )

    @staticmethod
    def _run_tokenization(text, stops, split, mark, pad_value):
        """
        Tokenizes a char level.
        """
        begin = 1 if mark else 0
        res = []
        if len(text.shape) == 1:
            for i in range(text.shape[0]):
                row = [pad_value for _ in range(begin)]
                for c in split(text[i]):
                    if c not in stops:
                        row.append(c)
                if mark:
                    row.append(pad_value)
                res.append(row)
            max_pos = max(map(len, res))
            for row in res:
                while len(row) < max_pos:
                    row.append(pad_value)
            res = np.array(res)
        elif len(text.shape) == 2:
            max_pos = 0
            for i in range(text.shape[0]):
                row2 = []
                for ii in range(text.shape[1]):
                    row = [pad_value for _ in range(begin)]
                    for c in split(text[i, ii]):
                        if c not in stops:
                            row.append(c)
                    if mark:
                        row.append(pad_value)
                    max_pos = max(max_pos, len(row))
                    row2.append(row)
                res.append(row2)
            for row2 in res:
                for row in row2:
                    while len(row) < max_pos:
                        row.append(pad_value)
            res = np.array(res)
        else:
            raise RuntimeError(
                f"Only vector or matrices are supported not shape {text.shape}."
            )
        return (res,)

    @staticmethod
    def _run_char_tokenization(text, stops, mark, pad_value):
        """
        Tokenizes by charaters.
        """

        def split(t):
            yield from t

        return Tokenizer._run_tokenization(text, stops, split, mark, pad_value)

    @staticmethod
    def _run_sep_tokenization(text, stops, separators, mark, pad_value):
        """
        Tokenizes using separators (as regular expressions).
        The function should use a trie to find text.
        """

        def split(t):
            begin = 0
            pos = 0
            while pos < len(t):
                for sep in separators:
                    if isinstance(sep, str):
                        if pos + len(sep) <= len(t) and sep == t[pos : pos + len(sep)]:
                            word = t[begin:pos]
                            yield word
                            begin = pos + len(sep)
                            break
                    else:
                        se = sep.match(t[pos:])
                        if se:
                            sep = se.group(0)
                            word = t[begin:pos]
                            yield word
                            begin = pos + len(sep)
                            break
                pos += 1
            if begin < pos:
                word = t[begin:pos]
                yield word

        return Tokenizer._run_tokenization(text, stops, split, mark, pad_value)

    @staticmethod
    def _run_regex_tokenization(text, stops, exp, tokenexpsplit, mark, pad_value):
        """
        Tokenizes using a regular expression.
        """
        if tokenexpsplit:

            def split(t):
                return filter(lambda x: x, exp.split(t))

        else:

            def split(t):
                return filter(lambda x: x, exp.findall(t))

        return Tokenizer._run_tokenization(text, stops, split, mark, pad_value)
