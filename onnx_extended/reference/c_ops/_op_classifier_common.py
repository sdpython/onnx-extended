import numpy


class _ClassifierCommon:
    """
    Labels strings are not natively implemented in C++ runtime.
    The class stores the strings labels, replaces them by
    integer, calls the C++ codes and then replaces them by strings.
    """

    @staticmethod
    def _post_process_predicted_label(label, scores, classlabels_int64s_string):
        """
        Replaces int64 predicted labels by the corresponding
        strings.
        """
        if classlabels_int64s_string is not None and len(classlabels_int64s_string) > 0:
            new_label = []
            no_array = False
            for i in label:
                if i >= len(classlabels_int64s_string):
                    new_label.append(None)
                    no_array = True
                else:
                    new_label.append(classlabels_int64s_string[i])
            if no_array:
                return new_label, scores
            return numpy.array(new_label), scores
        return label, scores
