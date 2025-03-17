def combine_by_frequency(mode_labels, omegas, target_binsize=4):
    """
    Inputs:



    Outputs: a list of lists describing the combined modes in terms of the physical modes:
        e.g., [[`mode1`, `mode2`, `mode6`], [`mode3`, `mode4`, `mode5`]]

        if `mode1`, `mode2`, and `mode6` were combined into a logical mode
        and same with 3 remaining

    """

    frequency_dict = {mode: omega for mode, omega in zip(mode_labels, omegas)}
    frequency_dict = dict(sorted(frequency_dict.items(), key=lambda item: item[1], reverse=True))

    bins = []
    current_bin = []

    for mode, freq in frequency_dict.items():
        if len(current_bin) >= target_binsize:
            bins.append(current_bin)
            current_bin = []
        current_bin.append(mode)

    if current_bin:
        bins.append(current_bin)

    return bins
