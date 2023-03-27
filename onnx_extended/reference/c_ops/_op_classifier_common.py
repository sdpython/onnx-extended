import numpy


class _ClassifierCommon:
    """
    Labels strings are not natively implemented in C++ runtime.
    The class stores the strings labels, replaces them by
    integer, calls the C++ codes and then replaces them by strings.
    """

    def _post_process_predicted_label(self, label, scores):
        """
        Replaces int64 predicted labels by the corresponding
        strings.
        """
        if self._classlabels_int64s_string is not None:
            new_label = numpy.array([self._classlabels_int64s_string[i] for i in label])
            return new_label, scores
        return label, scores
